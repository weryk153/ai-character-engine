/**
 * Minimal host-side adapter for VRMRendererPacket (VRM 0.x + 1.x).
 *
 * Verified against the current @pixiv/three-vrm v3 API shape used by its
 * official examples: expressionManager.setValue(),
 * humanoid.getNormalizedBoneNode(), and VRM.update().
 *
 * The engine intentionally does not own your AnimationMixer or animation asset
 * registry. Pass those in from the renderer host.
 */
import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';

type Command = {
  op: string;
  target: string;
  channel: string;
  duration_ms: number;
  start_offset_ms: number;
  values: Record<string, number | string | boolean>;
  source: string;
};

type Packet = {
  schema_version: number;
  sequence: number;
  source_event: string;
  model: { name: string; sha256: string; spec_version?: string; spec_family?: 'vrm0' | 'vrm1' };
  commands: Command[];
  reset: string[];
};

export type RendererContext = {
  vrm: VRM;
  animationActions: Map<string, THREE.AnimationAction>;
  // Track the actual names that were applied so reset works for both VRM 0.x
  // groups (for example A/Joy/Blink) and VRM 1.x presets (aa/happy/blink).
  activeExpressions?: Set<string>;
  expressionChannels?: Map<string, string>;
};

const rad = THREE.MathUtils.degToRad;

function resolveExpressionName(vrm: VRM, command: Command): string {
  const manager = vrm.expressionManager as unknown as {
    getExpression?: (name: string) => unknown;
  } | null;
  const logical = typeof command.values.logical_expression === 'string'
    ? command.values.logical_expression
    : undefined;
  if (manager?.getExpression) {
    if (manager.getExpression(command.target)) return command.target;
    if (logical && manager.getExpression(logical)) return logical;
  }
  return command.target;
}

export function applyRendererPacket(ctx: RendererContext, packet: Packet): string[] {
  const warnings: string[] = [];
  const { vrm } = ctx;

  for (const reset of packet.reset) {
    if (reset.startsWith('expressions:')) {
      const active = ctx.activeExpressions ?? new Set<string>();
      const channel = reset.slice('expressions:'.length);
      for (const name of [...active]) {
        const appliedChannel = ctx.expressionChannels?.get(name);
        if (appliedChannel && appliedChannel !== channel) continue;
        vrm.expressionManager?.setValue(name, 0);
        active.delete(name);
        ctx.expressionChannels?.delete(name);
      }
      ctx.activeExpressions = active;
    } else if (reset === 'look_at') {
      if (vrm.lookAt) {
        vrm.lookAt.autoUpdate = false;
        vrm.lookAt.yaw = 0;
        vrm.lookAt.pitch = 0;
      }
    } else if (reset.startsWith('humanoid:')) {
      const bone = reset.slice('humanoid:'.length);
      vrm.humanoid.getNormalizedBoneNode(bone as never)?.rotation.set(0, 0, 0);
    } else if (reset.startsWith('animation:')) {
      for (const action of ctx.animationActions.values()) action.stop();
    }
  }

  for (const command of packet.commands) {
    if (command.op === 'set_expression') {
      const name = resolveExpressionName(vrm, command);
      vrm.expressionManager?.setValue(name, Number(command.values.weight ?? 0));
      (ctx.activeExpressions ??= new Set<string>()).add(name);
      const channel = command.channel === 'mouth' ? 'mouth'
        : name.toLowerCase().startsWith('blink') ? 'blink' : 'face';
      (ctx.expressionChannels ??= new Map<string, string>()).set(name, channel);
      continue;
    }

    if (command.op === 'rotate_humanoid_delta') {
      const node = vrm.humanoid.getNormalizedBoneNode(command.target as never);
      if (!node) {
        warnings.push(`missing humanoid bone: ${command.target}`);
        continue;
      }
      node.rotation.set(
        rad(Number(command.values.pitch_delta_deg ?? 0)),
        rad(Number(command.values.yaw_delta_deg ?? 0)),
        rad(Number(command.values.roll_delta_deg ?? 0)),
      );
      continue;
    }

    if (command.op === 'look_at_angles') {
      if (vrm.lookAt) {
        // Packet angles own gaze until the host explicitly restores target tracking.
        // Public setters keep three-vrm's cached state and the eye bones in sync.
        vrm.lookAt.autoUpdate = false;
        vrm.lookAt.yaw = Number(command.values.yaw_deg ?? 0);
        vrm.lookAt.pitch = Number(command.values.pitch_deg ?? 0);
      } else {
        warnings.push('model has no lookAt controller');
      }
      continue;
    }

    if (command.op === 'play_animation') {
      const action = ctx.animationActions.get(command.target);
      if (!action) {
        warnings.push(`animation action not registered: ${command.target}`);
        continue;
      }
      action.reset().setEffectiveWeight(Number(command.values.weight ?? 1)).play();
      continue;
    }

    warnings.push(`unsupported renderer op: ${command.op}`);
  }

  vrm.expressionManager?.update();
  vrm.update(0);
  return warnings;
}
