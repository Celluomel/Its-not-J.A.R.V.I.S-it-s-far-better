import { Component, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

const vertex = `
uniform float time; uniform float energy;
varying vec3 vPosition; varying vec3 vView;
void main() {
  vec3 n = normalize(position);
  float angle = atan(n.z,n.x);
  // Broad radial creases create the folded membrane silhouette; the smaller
  // wave keeps the surface alive without turning it into a regular sphere.
  float foldBand = sin(angle*5.0+n.y*4.0+time*.23)*.22
                 + sin(angle*9.0-n.y*6.0-time*.16)*.08;
  float edge = smoothstep(.05,.95,abs(n.y));
  float radius = 1.0 + foldBand*(.48 + edge*.34)
               + sin(n.y*8.0+angle*3.0+time*.4)*.035;
  vec3 p = position;
  p.x *= radius*1.18;
  p.z *= radius*.64;
  p.y *= .82;
  p.y += sin(angle*5.0+n.y*4.0+time*.23)*.12*(.35+edge);
  p.z += cos(angle*7.0-n.y*3.0-time*.16)*.07;
  p.y += sin(time*.65)*.035*(1.0+energy);
  vPosition = p;
  vec4 view = modelViewMatrix*vec4(p,1.0);
  vView = view.xyz;
  gl_Position = projectionMatrix*view;
}`;
const fragment = `
uniform float time; uniform float energy;
varying vec3 vPosition; varying vec3 vView;
void main() {
  vec3 n = normalize(cross(dFdx(vView),dFdy(vView)));
  float facing = abs(dot(n,normalize(-vView)));
  float rim = pow(1.0-facing,2.4);
  float fresnel = pow(1.0-facing,3.0);
  float silk = pow(abs(dot(n,normalize(vec3(-.6,.8,1.0)))),12.0);
  float reflection = pow(max(0.0,dot(n,normalize(vec3(-.4,.7,1.0)))),22.0);
  float warmth = smoothstep(-1.0,1.0,vPosition.x-vPosition.y*.45);
  // Layered moving caustics make the highlights slide like reflections on
  // dark water instead of behaving like a fixed metallic sheen.
  float waveA = sin(vPosition.x*8.0 + vPosition.z*11.0 + time*1.35);
  float waveB = sin(vPosition.y*13.0 - vPosition.x*5.0 - time*.92);
  float waterCaustic = pow(max(0.0, waveA*waveB), 3.0);
  float waterSheen = pow(max(0.0, sin(vPosition.z*18.0 + vPosition.x*4.0 + time*1.1)), 8.0);
  // Thin-film interference shifts through teal, rose, amber and violet as
  // the folded surface turns toward the viewer, like iridescent glass.
  float phase = vPosition.x*1.7 + vPosition.y*2.2 + vPosition.z*3.1 + fresnel*4.5;
  vec3 spectral = .5 + .5*cos(vec3(phase, phase+2.1, phase+4.2));
  spectral = mix(vec3(.20,.78,.70), spectral, .58);
  vec3 pearl = mix(vec3(.25,.70,.66),vec3(.95,.66,.58),warmth);
  vec3 color = mix(pearl, spectral, .28 + fresnel*.62);
  color += vec3(1.0,.72,.58)*reflection*1.8;
  color += vec3(.72,1.0,.94)*silk*.72;
  color += vec3(.24,.85,.82)*waterCaustic*(.45 + energy*.75);
  color += vec3(1.0,.86,.74)*waterSheen*(.35 + fresnel*.8);
  gl_FragColor = vec4(color,.018+rim*.46+fresnel*.24+silk*.12+waterCaustic*.07);
}`;
const pointVertex = `
uniform float time; uniform float energy;
attribute float size; attribute vec3 tint;
varying vec3 vTint;
void main(){
  vec3 p=position;
  p.x+=sin(p.y*3.0+time*.3)*.025;
  p.z+=cos(p.x*4.0+time*.23)*.025;
  p*=1.0+sin(time*.7)*.018;
  vTint=tint;
  vec4 view=modelViewMatrix*vec4(p,1.0);
  gl_Position=projectionMatrix*view;
  gl_PointSize=clamp(size*(150.0/-view.z)*(1.0+energy*.2),1.0,5.0);
}`;
const pointFragment = `
varying vec3 vTint;
void main(){
  float d=length(gl_PointCoord-.5)*2.0;
  if(d>1.0) discard;
  gl_FragColor=vec4(vTint,pow(1.0-d,1.5)*.85);
}`;
const heartFragment = `
varying vec3 vPosition; varying vec3 vView;
uniform float time; uniform vec3 coreColor; uniform float coreIntensity;
void main(){
  vec3 n=normalize(cross(dFdx(vView),dFdy(vView)));
  float facing=abs(dot(n,normalize(-vView)));
  float folds=sin(vPosition.y*22.0+vPosition.x*14.0+time*.5)*.5+.5;
  vec3 red=mix(coreColor*.72,coreColor*1.35,folds);
  gl_FragColor=vec4(red*(.72+coreIntensity*.55),.16+facing*.38+coreIntensity*.08);
}`;

function Atmosphere({ energy, reduced }: { energy: number; reduced: boolean }) {
  const points = useRef<THREE.Points>(null);
  const geometry = useMemo(() => {
    const positions: number[] = [];
    let seed = 421;
    const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    for (let i = 0; i < 180; i++) {
      positions.push((random() - .5) * 7, (random() - .5) * 4.8, -1.8 - random() * 2.4);
    }
    const next = new THREE.BufferGeometry();
    next.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    return next;
  }, []);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useFrame((state, delta) => {
    if (!points.current || reduced) return;
    points.current.rotation.z += delta * (.003 + energy * .006);
    points.current.position.x = Math.sin(state.clock.elapsedTime * .08) * .025;
  });
  return <points ref={points} geometry={geometry} renderOrder={-2}>
    <pointsMaterial color="#9acfc4" size={.018} transparent opacity={.16 + energy * .10} depthWrite={false} sizeAttenuation/>
  </points>;
}

function SculpturalLighting({ energy, reduced }: { energy: number; reduced: boolean }) {
  const key = useRef<THREE.SpotLight>(null);
  const rim = useRef<THREE.SpotLight>(null);
  useFrame((state, delta) => {
    if (reduced) return;
    const t = state.clock.elapsedTime;
    if (key.current) {
      key.current.position.x = -2.5 + Math.sin(t * .18) * .35;
      key.current.position.y = 2.5 + Math.cos(t * .15) * .18;
      key.current.intensity = 3.4 + energy * .9;
    }
    if (rim.current) {
      rim.current.position.x = 2.4 + Math.cos(t * .14) * .4;
      rim.current.position.y = .5 + Math.sin(t * .2) * .25;
      rim.current.intensity = 2.8 + energy * 1.2;
    }
  });
  return <>
    <spotLight ref={key} position={[-2.5,2.5,3]} angle={.42} penumbra={.82} decay={1.5} distance={8} intensity={3.8} color="#ffd1bd"/>
    <spotLight ref={rim} position={[2.4,.5,1.4]} angle={.5} penumbra={.9} decay={1.4} distance={7} intensity={3.2} color="#63e6d0"/>
    <spotLight position={[0,-2.5,2]} angle={.62} penumbra={1} decay={1.7} distance={7} intensity={1.7} color="#a9a4ff"/>
  </>;
}

const shadowVertex = `varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
const shadowFragment = `varying vec2 vUv; uniform float energy;
void main(){ float d=length((vUv-.5)*2.); float softness=pow(max(0.,1.-d),2.0); float inner=pow(max(0.,1.-d*.72),3.0); float rim=pow(max(0.,1.-abs(d-.62)*4.0),2.0); vec3 color=mix(vec3(.10,.34,.32),vec3(.34,.20,.30),rim); gl_FragColor=vec4(color,softness*(.34+energy*.18)+inner*.14); }`;

function GroundShadow({ energy, reduced }: { energy: number; reduced: boolean }) {
  const material = useRef<THREE.ShaderMaterial>(null);
  const mesh = useRef<THREE.Mesh>(null);
  useFrame((state, delta) => {
    if (!material.current || reduced) return;
    material.current.uniforms.energy.value = energy;
    const scale = 1 + Math.sin(state.clock.elapsedTime * .7) * (.018 + energy * .012);
    mesh.current?.scale.set(1.18 * scale, .34 / scale, 1);
  });
  return <mesh ref={mesh} position={[0, -1.58, .18]} scale={[1.18, .34, 1]} rotation={[0, 0, 0]} renderOrder={-1}>
    <circleGeometry args={[1, 96]}/>
    <shaderMaterial ref={material} vertexShader={shadowVertex} fragmentShader={shadowFragment} uniforms={{ energy: { value: energy } }} transparent depthWrite={false}/>
  </mesh>;
}

type OrganismMode = 'idle' | 'listening' | 'processing' | 'responding';
function Body({ energy, reduced, mode }: { energy: number; reduced: boolean; mode: OrganismMode }) {
  const viewport = useThree(state => state.viewport);
  const frameScale = Math.min(1, viewport.width / 4.1, viewport.height / 4.1);
  const group = useRef<THREE.Group>(null);
  const material = useRef<THREE.ShaderMaterial>(null);
  const uniforms = useMemo(() => ({ time: { value: 0 }, energy: { value: 0 }, coreColor: { value: new THREE.Color('#ed392d') }, coreIntensity: { value: .35 } }), []);
  const { particles, filaments } = useMemo(() => {
    let seed = 83;
    const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    const positions: number[] = [], colors: number[] = [], sizes: number[] = [], lines: number[] = [];
    for (let i = 0; i < 2600; i++) {
      const azimuth = random()*Math.PI*2;
      const y = random()*2-1;
      const radius = Math.pow(random(),.43)*1.22;
      const horizontal = Math.sqrt(1-y*y)*radius;
      positions.push(Math.cos(azimuth)*horizontal,y*radius,Math.sin(azimuth)*horizontal);
      const tint = new THREE.Color(radius < .53 ? '#ff4934' : random() > .5 ? '#8edec8' : '#e7bd9c');
      colors.push(tint.r,tint.g,tint.b);
      sizes.push(.025+Math.pow(random(),3)*.095);
    }
    // Radial capillaries connect the warm core to scattered internal points.
    for (let branch = 0; branch < 55; branch++) {
      const end = new THREE.Vector3(...positions.slice(branch*3,branch*3+3) as [number,number,number]);
      const phase = random()*6.28;
      const point = (t: number) => new THREE.Vector3(end.x*t+Math.sin(t*9+phase)*t*.07,end.y*t,end.z*t+Math.cos(t*8+phase)*t*.07);
      for (let i=0;i<25;i++) lines.push(...point(i/25).toArray(),...point((i+1)/25).toArray());
    }
    const particles = new THREE.BufferGeometry();
    particles.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
    particles.setAttribute('tint',new THREE.Float32BufferAttribute(colors,3));
    particles.setAttribute('size',new THREE.Float32BufferAttribute(sizes,1));
    const filaments = new THREE.BufferGeometry();
    filaments.setAttribute('position',new THREE.Float32BufferAttribute(lines,3));
    return { particles, filaments };
  }, []);
  useEffect(() => () => { particles.dispose(); filaments.dispose(); }, [particles, filaments]);
  useFrame((state, delta) => {
    if (!group.current || !material.current) return;
    const dt = Math.min(delta, .05);
    material.current.uniforms.energy.value = THREE.MathUtils.damp(material.current.uniforms.energy.value, energy, 4, dt);
    const colors: Record<OrganismMode, string> = { idle: '#ed392d', listening: '#61d7df', processing: '#f2b45f', responding: '#91e6ad' };
    material.current.uniforms.coreColor.value.lerp(new THREE.Color(colors[mode]), 1 - Math.exp(-5 * dt));
    material.current.uniforms.coreIntensity.value = THREE.MathUtils.damp(material.current.uniforms.coreIntensity.value, mode === 'idle' ? .35 : 1, 5, dt);
    if (!reduced) {
      material.current.uniforms.time.value += dt;
      group.current.rotation.y += dt * .035;
      group.current.rotation.z = Math.sin(state.clock.elapsedTime * .12) * .08;
      const size = 1 + Math.sin(state.clock.elapsedTime * .7) * .018;
      group.current.scale.setScalar(size);
      group.current.rotation.x = THREE.MathUtils.damp(group.current.rotation.x, state.pointer.y * .12, 2, dt);
    }
  });
  return <group scale={frameScale}><group ref={group} rotation={[.15, .3, .1]}>
    <mesh scale={[1.05,.92,1]} renderOrder={4}><sphereGeometry args={[1.4, 144, 96]} /><shaderMaterial ref={material} vertexShader={vertex} fragmentShader={fragment} uniforms={uniforms} transparent side={THREE.DoubleSide} depthWrite={false}/></mesh>
    <mesh scale={[.92,.86,.82]} rotation={[.1,.38,.12]} renderOrder={3}><sphereGeometry args={[1.36,112,80]}/><shaderMaterial vertexShader={vertex} fragmentShader={fragment} uniforms={uniforms} transparent side={THREE.DoubleSide} depthWrite={false}/></mesh>
    <mesh scale={[.76,.72,.66]} rotation={[.3,-.5,-.2]} renderOrder={2}><sphereGeometry args={[1.3,96,64]}/><shaderMaterial vertexShader={vertex} fragmentShader={fragment} uniforms={uniforms} transparent side={THREE.DoubleSide} depthWrite={false}/></mesh>
    <points geometry={particles} renderOrder={1}><shaderMaterial vertexShader={pointVertex} fragmentShader={pointFragment} uniforms={uniforms} transparent blending={THREE.AdditiveBlending} depthWrite={false}/></points>
    <lineSegments geometry={filaments}><lineBasicMaterial color="#ef8168" transparent opacity={.16} depthWrite={false}/></lineSegments>
    <mesh scale={[.26,.38,.26]}><sphereGeometry args={[1,64,48]}/><shaderMaterial vertexShader={vertex} fragmentShader={heartFragment} uniforms={uniforms} transparent side={THREE.DoubleSide} depthWrite={false}/></mesh>
    <mesh scale={[.36,.24,.32]} rotation={[.5,0,.7]}><sphereGeometry args={[1,48,40]}/><shaderMaterial vertexShader={vertex} fragmentShader={heartFragment} uniforms={uniforms} transparent side={THREE.DoubleSide} depthWrite={false}/></mesh>
  </group></group>;
}

class GraphicsBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <div className="graphics-fallback">PandoraBOX<span>Graphics unavailable</span></div> : this.props.children; }
}

export default function Organism({ active, energy = 0, reduced, mode = active ? 'processing' : 'idle' }: { active: boolean; energy?: number; reduced: boolean; mode?: OrganismMode }) {
  const [visible, setVisible] = useState(!document.hidden);
  const [lost, setLost] = useState(false);
  useEffect(() => { const change = () => setVisible(!document.hidden); document.addEventListener('visibilitychange', change); return () => document.removeEventListener('visibilitychange', change); }, []);
  return <div className="organism" role="img" aria-label={`PandoraBOX visual organism, ${mode}`}>
    {lost ? <div className="graphics-fallback">PandoraBOX<span>Graphics paused. Reload to restore.</span></div> : <GraphicsBoundary><Canvas camera={{ position: [0, 0, 4.9], fov: 43 }} dpr={[1, 1.5]} frameloop={!visible ? 'never' : reduced ? 'demand' : 'always'} gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }} onCreated={({gl, scene}) => { gl.toneMappingExposure = 1.12; scene.background = new THREE.Color('#060d12'); scene.fog = new THREE.Fog('#060d12', 4.5, 8); gl.domElement.addEventListener('webglcontextlost', () => setLost(true), { once: true }); }}><ambientLight intensity={.18}/><SculpturalLighting energy={active ? .65 : energy} reduced={reduced}/><Atmosphere energy={active ? .65 : energy} reduced={reduced}/><GroundShadow energy={active ? .65 : energy} reduced={reduced}/><Body energy={active ? .65 : energy} mode={mode} reduced={reduced}/></Canvas></GraphicsBoundary>}
  </div>;
}
