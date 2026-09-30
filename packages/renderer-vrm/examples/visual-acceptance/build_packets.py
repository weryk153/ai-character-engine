import argparse
import json
from pathlib import Path
from ai_character_engine.avatar import AvatarCueBundle, VisemeCue, ExpressionCue, AvatarBehaviorCueBundle, AvatarBehaviorPhase, GazeCue,GazeTarget,HeadMotionCue
from ai_character_engine.live import LiveRuntimeEvent,LiveEventType
from ai_character_engine_vrm import inspect_vrm_path,VRMRendererBridge
parser=argparse.ArgumentParser();parser.add_argument('model',type=Path);args=parser.parse_args()
b=VRMRendererBridge(inspect_vrm_path(args.model));p={}
def add(name,kind,data):p[name]=b.compile_event(LiveRuntimeEvent(kind,data=data)).to_dict()
for k in 'AIUEO':add(k,LiveEventType.AVATAR_CUE,AvatarCueBundle('acceptance',0,0,0,1000,visemes=(VisemeCue(k.lower(),0,1000),)).to_dict())
for name,logical in [('Blink','blink'),('Blink_L','blinkLeft'),('Blink_R','blinkRight'),('Joy','happy'),('Sorrow','sad'),('Angry','angry'),('Surprised','surprised')]:add(name,LiveEventType.AVATAR_CUE,AvatarCueBundle('acceptance',0,0,0,1000,expressions=(ExpressionCue(logical,0,1000,group='blink' if name.startswith('Blink') else 'face'),)).to_dict())
for name,yaw,pitch in [('Look left',-20,0),('Look right',20,0),('Look up',0,15),('Look down',0,-15)]:add(name,LiveEventType.AVATAR_BEHAVIOR_CUE,AvatarBehaviorCueBundle(0,AvatarBehaviorPhase.IDLE,0,gaze=GazeCue(GazeTarget(name,yaw,pitch),1000)).to_dict())
for axis in ['yaw','pitch','roll']:
 for sign in [-1,1]:
  a=dict(yaw_delta_deg=0,pitch_delta_deg=0,roll_delta_deg=0,duration_ms=1000);a[axis+'_delta_deg']=sign*10
  add('Head '+axis+(' +' if sign>0 else ' −'),LiveEventType.AVATAR_BEHAVIOR_CUE,AvatarBehaviorCueBundle(0,AvatarBehaviorPhase.IDLE,0,head_motion=HeadMotionCue(**a)).to_dict())
add('Reset expressions',LiveEventType.AVATAR_RESET,{'reset':['mouth','expressions']})
add('Reset behavior',LiveEventType.AVATAR_BEHAVIOR_RESET,{'reset':['gaze','blink','head','idle_motion','gestures']})
Path(__file__).with_name('packets.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
print({k:v['warnings'] for k,v in p.items() if v['warnings']})
