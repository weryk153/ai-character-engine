import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import {VRMLoaderPlugin,VRMUtils} from '@pixiv/three-vrm';
import {applyRendererPacket} from '../three_vrm_packet_adapter';
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(Math.min(devicePixelRatio,2));document.body.append(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#e7eef2');scene.add(new THREE.HemisphereLight(0xffffff,0x788899,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(1,2,3);scene.add(light);
const camera=new THREE.PerspectiveCamera(28,innerWidth/innerHeight,.01,100);camera.position.set(0,1.35,2.9);camera.lookAt(-.2,1.1,0);
const loader=new GLTFLoader();loader.register(parser=>new VRMLoaderPlugin(parser));const gltf=await loader.loadAsync('./model.vrm');const vrm=gltf.userData.vrm;VRMUtils.rotateVRM0(vrm);scene.add(vrm.scene);
const packets=await (await fetch('./packets.json')).json();const ctx={vrm,animationActions:new Map()};let spring=false,frame=0,prev=performance.now(),maxFrame=0;const frameTimes:number[]=[];const evidence:any[]=[];
function state(){const bones:any={};for(const key of ['head','leftEye','rightEye']){const b=vrm.humanoid.getRawBoneNode(key);bones[key]=b?.quaternion.toArray();}let nonfinite=0;vrm.scene.traverse(o=>{if(!o.matrixWorld.elements.every(Number.isFinite))nonfinite++;});return {bones,lookAt:{yaw:vrm.lookAt?.yaw,pitch:vrm.lookAt?.pitch},expressions:Object.fromEntries(vrm.expressionManager.expressions.map(e=>[e.expressionName,e.weight])),nonfinite,frame,maxFrame,webgl:renderer.info.render,geometries:renderer.info.memory.geometries,textures:renderer.info.memory.textures,springJoints:vrm.springBoneManager?.joints?.size};}
function apply(name:string){const warnings=applyRendererPacket(ctx,packets[name]);const r={name,warnings,state:state(),human_confirmed:false};evidence.push(r);document.querySelector('#status')!.textContent=JSON.stringify(r,null,2);return r;}
function reset(){spring=false;vrm.scene.position.x=0;apply('Reset expressions');apply('Reset behavior');}
for(const name of Object.keys(packets)){const button=document.createElement('button');button.textContent=name;button.onclick=()=>{reset();apply(name);};document.querySelector('#buttons')!.append(button);}
(document.querySelector('#reset') as HTMLElement).onclick=reset;(document.querySelector('#spring') as HTMLElement).onclick=()=>{reset();spring=true;};
function animate(now:number){requestAnimationFrame(animate);const dt=Math.min((now-prev)/1000,.05);maxFrame=Math.max(maxFrame,now-prev);if(frameTimes.length<50000)frameTimes.push(now-prev);prev=now;frame++;if(spring){vrm.scene.position.x=.08*Math.sin(now/300);const head=vrm.humanoid.getNormalizedBoneNode('head');head.rotation.set(.12*Math.sin(now/500),.28*Math.sin(now/380),.1*Math.sin(now/470));}vrm.update(dt);renderer.render(scene,camera);}requestAnimationFrame(animate);
(window as any).acceptance={ready:true,apply,reset,state,evidence,packets,vrm,ctx,camera,setSpring:(v:boolean)=>spring=v,frameTimes};reset();
