import { Component, useEffect, useMemo, useRef, useState, type MutableRefObject, type ReactNode } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export const networkColors = ['#72ead4', '#ffac86', '#92b9ff', '#e5c678', '#e9eff2', '#b6a1ef'];
const workspaceColor = '#ffd0a8';
export type WorkspaceSignal = { focus?: string; confidence?: number | null; active_hypotheses?: unknown[] };
export type SculptureProps = { levels: (number | null)[]; selected: number; onSelect: (index: number) => void; moving: boolean; reset: number; live: boolean; workspace?: WorkspaceSignal };

// A folded annulus: six interleaved cortical ribbons, each with a distinct signal path.
function surface(u: number, v: number, layer: number) {
  const a = u * Math.PI * 2;
  const b = v * Math.PI * 2 + layer * Math.PI / 3 + .65 * Math.sin(a * 2);
  const major = 1.27 + .17 * Math.sin(a * 3 + .4);
  const minor = .83 + .14 * Math.cos(a * 3 + .7);
  const fold = .028 * Math.sin(a * 32 + b * 2);
  const r = major + (minor + fold) * Math.cos(b);
  return new THREE.Vector3(r * Math.cos(a), r * Math.sin(a) * 1.08, (minor + fold) * Math.sin(b) + .32 * Math.sin(a * 3));
}

function ribbonGeometry(layer: number) {
  const positions: number[] = [], uvs: number[] = [], indices: number[] = [];
  const around = 280, across = 14;
  for (let i = 0; i <= around; i++) {
    for (let j = 0; j <= across; j++) {
      const u = i / around, v = (j / across - .5) * .134 + .048 * Math.sin(u * Math.PI * 4);
      positions.push(...surface(u, v, layer).toArray());
      uvs.push(u, j / across);
      if (i < around && j < across) {
        const p = i * (across + 1) + j;
        indices.push(p, p + across + 1, p + 1, p + 1, p + across + 1, p + across + 2);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

const flowVertex = `varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
const flowFragment = `
uniform float time; uniform float level; uniform vec3 tint; varying vec2 vUv;
void main(){
  float travel=fract(vUv.x*3.-time*(.07+level*.14));
  float pulse=pow(max(0.,1.-travel),19.);
  vec3 c=tint*(.32+level*.42+pulse*3.1*level);
  gl_FragColor=vec4(c,1.);
}`;

function lobeGeometry(index: number) {
  const geometry = new THREE.IcosahedronGeometry(.42, 3);
  const position = geometry.getAttribute('position');
  const vertex = new THREE.Vector3();
  for (let i = 0; i < position.count; i++) {
    vertex.fromBufferAttribute(position, i);
    const ripple = 1 + .11 * Math.sin(vertex.x * 11 + vertex.y * 7 + index * 1.9)
      + .045 * Math.cos(vertex.z * 17 - index);
    vertex.multiplyScalar(ripple);
    position.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  position.needsUpdate = true;
  geometry.computeVertexNormals();
  return geometry;
}

function Cortex({ selected, levels, onSelect, moving, live, workspace }: SculptureProps) {
  const group = useRef<THREE.Group>(null);
  const elapsed = useRef(0);
  const viewport = useThree(state => state.viewport);
  const scale = Math.min(1, viewport.width / 5.55, viewport.height / 5.6);
  const model = useMemo(() => {
    return networkColors.map((color, layer) => {
      const points = Array.from({ length: 281 }, (_, i) => surface(i / 280, .068 + .048 * Math.sin(i / 280 * Math.PI * 4), layer));
      const curve = new THREE.CatmullRomCurve3(points, true);
      const flow = new THREE.ShaderMaterial({
        vertexShader: flowVertex, fragmentShader: flowFragment,
        uniforms: { time: { value: 0 }, level: { value: 0 }, tint: { value: new THREE.Color(color) } },
      });
      return { ribbon: ribbonGeometry(layer), conduit: new THREE.TubeGeometry(curve, 320, .010, 5, true), flow };
    });
  }, []);
  const activity = levels.reduce<number>((sum, value) => sum + (value ?? 0), 0) / Math.max(1, levels.length);
  useEffect(() => () => model.forEach(item => { item.ribbon.dispose(); item.conduit.dispose(); item.flow.dispose(); }), [model]);
  useFrame((_, delta) => {
    const dt = Math.min(delta, .05);
    // A living sculpture needs visible motion at a glance. Keep it calm, but
    // let activity accelerate the internal time instead of freezing at 1x.
    if (moving) elapsed.current += dt * (1.65 + activity * 1.35);
    model.forEach((item, i) => {
      item.flow.uniforms.time.value = elapsed.current;
      // The central heart owns Global Workspace; this legacy layer now carries
      // the independent reasoning-confidence signal.
      const ribbonLevel = live ? levels[i] ?? 0 : 0;
      item.flow.uniforms.level.value = THREE.MathUtils.damp(item.flow.uniforms.level.value, ribbonLevel, 3, dt);
    });
    if (group.current) {
      group.current.rotation.y = .66 + Math.sin(elapsed.current * .22) * .22;
      group.current.rotation.x = -.19 + Math.sin(elapsed.current * .31) * .06;
      group.current.rotation.z = -.23 + Math.sin(elapsed.current * .17) * .035;
      group.current.position.y = Math.sin(elapsed.current * .72) * .045;
    }
  });
  return <group scale={scale}><group ref={group} rotation={[-.19, .42, -.23]}>
    {model.map((item, i) => <group key={i}>
      <mesh geometry={item.ribbon} onClick={e => { e.stopPropagation(); onSelect(i); }}>
        <meshPhysicalMaterial color={i === selected && i !== 2 ? networkColors[i] : i === 2 ? '#526b6b' : i % 3 === 0 ? '#7a9996' : '#b0bab4'} metalness={.65} roughness={.32} envMapIntensity={.55} clearcoat={.7} clearcoatRoughness={.28} side={THREE.DoubleSide} emissive={networkColors[i]} emissiveIntensity={i === selected && i !== 2 ? .035 : .004}/>
      </mesh>
      <mesh geometry={item.conduit} material={item.flow}/>
    </group>)}
    <WorkspaceHeart confidence={workspace?.confidence} hypotheses={workspace?.active_hypotheses?.length ?? 0} selectedColor={selected === 2 ? workspaceColor : networkColors[selected]} moving={moving} live={live} elapsed={elapsed}/>
    <OrganicLobes selected={selected} levels={levels} onSelect={onSelect} moving={moving} live={live} elapsed={elapsed}/>
    <Filaments selected={selected} moving={moving} live={live} level={levels[selected] ?? 0}/>
  </group></group>;
}

function WorkspaceHeart({ confidence, hypotheses, selectedColor, moving, live, elapsed }: { confidence?: number | null; hypotheses: number; selectedColor: string; moving: boolean; live: boolean; elapsed: MutableRefObject<number> }) {
  const group = useRef<THREE.Group>(null);
  const confidenceLevel = Math.max(0, Math.min(1, confidence ?? 0));
  const hypothesisSignal = Math.min(1, hypotheses / 5);
  useFrame((_, delta) => {
    if (!group.current || !moving) return;
    const pulse = .5 + .5 * Math.sin(elapsed.current * (1.8 + confidenceLevel * 2.4));
    const target = 1 + pulse * (.035 + confidenceLevel * .08) + hypothesisSignal * .025;
    group.current.scale.lerp(new THREE.Vector3(target, target, target), Math.min(1, delta * 9));
    group.current.rotation.y += delta * (.16 + confidenceLevel * .28);
    group.current.rotation.x += delta * (.05 + hypothesisSignal * .08);
  });
  const intensity = live ? .15 + confidenceLevel * .55 + hypothesisSignal * .2 : .04;
  return <group ref={group} rotation={[.4, .1, .6]} userData={{ subsystem: 'global_workspace' }}>
    <pointLight color="#ffb08f" intensity={live ? 1.2 + confidenceLevel * 2.4 : .35} distance={2.6}/>
    <mesh>
      <torusKnotGeometry args={[.40, .135, 200, 24, 2, 3]}/>
      <meshPhysicalMaterial color="#e0a384" metalness={.83} roughness={.23} clearcoat={1} emissive="#c25132" emissiveIntensity={intensity}/>
    </mesh>
    <mesh scale={.7}>
      <icosahedronGeometry args={[.42, 3]}/>
      <meshPhysicalMaterial color="#ffd4bd" emissive="#ff7657" emissiveIntensity={live ? .28 + confidenceLevel * .7 : .06} metalness={.12} roughness={.16} transmission={.55} thickness={.7} transparent opacity={live ? .72 : .35}/>
    </mesh>
    <mesh rotation={[1.2, .3, 0]}>
      <torusGeometry args={[.65, .009, 6, 120]}/>
      <meshBasicMaterial color={selectedColor} toneMapped={false}/>
    </mesh>
  </group>;
}

function OrganicLobes({ selected, levels, onSelect, moving, live, elapsed }: Omit<SculptureProps, 'reset'> & { elapsed: MutableRefObject<number> }) {
  const group = useRef<THREE.Group>(null);
  const geometries = useMemo(() => networkColors.map((_, i) => lobeGeometry(i)), []);
  const placements = [
    [-.54, .62, .18], [.57, .48, .06], [-.76, -.18, .08], [.75, -.22, .12], [-.28, -.64, .1], [.34, -.66, -.06],
  ];
  useEffect(() => () => geometries.forEach(geometry => geometry.dispose()), [geometries]);
  useFrame((_, delta) => {
    if (!group.current || !moving) return;
    group.current.children.forEach((organ, index) => {
      const level = live ? levels[index] ?? 0 : 0;
      const beat = .5 + .5 * Math.sin(elapsed.current * (1.15 + level * 1.9) + index * 1.7);
      const selectedBoost = index === selected ? .07 : 0;
      const target = 1 + selectedBoost + beat * (.025 + level * .055);
      organ.scale.lerp(new THREE.Vector3(target, target * (1.02 + .03 * Math.sin(index)), target), Math.min(1, delta * 8));
      organ.rotation.y += delta * (.12 + level * .34);
      organ.rotation.z += delta * (.05 + level * .12);
    });
  });
  return <group ref={group}>
    {geometries.map((geometry, index) => {
      const [x, y, z] = placements[index];
      const level = levels[index] ?? 0;
      return <group key={index} position={[x, y, z]} rotation={[index * .31, index * .52, index * .17]} onClick={event => { event.stopPropagation(); onSelect(index); }}>
        <mesh geometry={geometry}>
          <meshPhysicalMaterial color={networkColors[index]} emissive={networkColors[index]} emissiveIntensity={index === selected ? .8 : .16 + level * .35} metalness={.28} roughness={.22} transmission={.38} thickness={.8} transparent opacity={index === selected ? .78 : .42} clearcoat={1} clearcoatRoughness={.18}/>
        </mesh>
        <mesh scale={.62}>
          <icosahedronGeometry args={[.42, 2]}/>
          <meshBasicMaterial color={networkColors[index]} transparent opacity={index === selected ? .18 : .055} blending={THREE.AdditiveBlending} depthWrite={false}/>
        </mesh>
      </group>;
    })}
  </group>;
}

function Filaments({ selected, moving, live, level }: { selected: number; moving: boolean; live: boolean; level: number }) {
  const pointsRef = useRef<THREE.Points>(null);
  const { geometry, fibers } = useMemo(() => {
    const positions: number[] = [], fibers: number[] = [];
    for (let i = 0; i < 90; i++) {
      const angle = i * 2.399963;
      const endpoint = surface((i * .618033) % 1, .03, i % 6);
      const start = new THREE.Vector3(Math.cos(angle) * .48, Math.sin(angle) * .48, Math.sin(i) * .25);
      const curve = new THREE.QuadraticBezierCurve3(start, new THREE.Vector3(endpoint.x * .7, endpoint.y * .65, Math.cos(i * .4) * 1.1), endpoint);
      const samples = curve.getPoints(26);
      samples.forEach((p, j) => { if (j < samples.length - 1) fibers.push(...p.toArray(), ...samples[j + 1].toArray()); });
      positions.push(...curve.getPoint((i * .31415) % 1).toArray());
    }
    return { geometry: new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)), fibers: new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(fibers, 3)) };
  }, []);
  useEffect(() => () => { geometry.dispose(); fibers.dispose(); }, [geometry, fibers]);
  useFrame((_, delta) => { if (pointsRef.current && moving && live) pointsRef.current.rotation.z += Math.min(delta, .05) * (.16 + .42 * level); });
  return <>
    <lineSegments geometry={fibers}><lineBasicMaterial color="#8db5ad" transparent opacity={.19}/></lineSegments>
    <points ref={pointsRef} geometry={geometry}><pointsMaterial color={networkColors[selected]} size={.022} transparent opacity={live ? .35 + level * .65 : .2} sizeAttenuation/></points>
  </>;
}

function Studio({ reset, moving }: { reset: number; moving: boolean }) {
  const { gl, scene, camera, size, invalidate } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  useEffect(() => {
    const room = new RoomEnvironment();
    const pmrem = new THREE.PMREMGenerator(gl);
    const environment = pmrem.fromScene(room, .04);
    scene.environment = environment.texture;
    scene.background = new THREE.Color('#0b1216');
    gl.toneMappingExposure = .85;
    const controls = new OrbitControls(camera, gl.domElement);
    controls.enablePan = false;
    controls.enableZoom = false;
    controls.enableDamping = moving;
    controls.minPolarAngle = Math.PI * .3;
    controls.maxPolarAngle = Math.PI * .7;
    const onControlsChange = () => invalidate();
    controls.addEventListener('change', onControlsChange);
    controlsRef.current = controls;
    const composer = new EffectComposer(gl);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), .12, .35, 1.5));
    composer.addPass(new OutputPass());
    composerRef.current = composer;
    invalidate();
    return () => {
      scene.environment = null;
      scene.background = null;
      controls.removeEventListener('change', onControlsChange);
      controls.dispose(); environment.dispose(); room.dispose(); pmrem.dispose();
      composer.passes.forEach(pass => pass.dispose()); composer.dispose();
      composerRef.current = null;
    };
  }, [gl, scene, camera, invalidate]);
  useEffect(() => { composerRef.current?.setSize(size.width, size.height); invalidate(); }, [size, invalidate]);
  useEffect(() => { if (controlsRef.current) controlsRef.current.enableDamping = moving; }, [moving]);
  useEffect(() => { camera.position.set(0, .25, 8.7); controlsRef.current?.target.set(0, 0, 0); controlsRef.current?.update(); invalidate(); }, [reset, camera, invalidate]);
  useFrame(() => { controlsRef.current?.update(); composerRef.current?.render(); }, 1);
  return <>
    <ambientLight intensity={.25}/>
    <directionalLight position={[-4, 5, 5]} intensity={1.8} color="#e6fff8"/>
    <directionalLight position={[4, 0, -3]} intensity={2.4} color="#55d9bf"/>
    <directionalLight position={[0, -3, 4]} intensity={.85} color="#ffad87"/>
    <spotLight position={[1, 5, 2]} intensity={15} angle={.65} penumbra={1}/>
  </>;
}

class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <div className="sculpture-fallback">3D rendering unavailable. Live telemetry remains below.</div> : this.props.children; }
}

export default function CognitiveSculpture(props: SculptureProps) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [lost, setLost] = useState(false);
  useEffect(() => {
    let intersecting = false;
    const update = () => setVisible(intersecting && !document.hidden);
    const observer = new IntersectionObserver(entries => { intersecting = entries[0].isIntersecting; update(); }, { rootMargin: '100px' });
    if (host.current) observer.observe(host.current);
    document.addEventListener('visibilitychange', update);
    return () => { observer.disconnect(); document.removeEventListener('visibilitychange', update); };
  }, []);
  return <div className="cognitive-sculpture" ref={host} role="img" aria-label="Interactive three-dimensional Lumina cognitive sculpture">
    {lost ? <div className="sculpture-fallback">Graphics context interrupted. Reload to restore the sculpture.</div> : <SceneBoundary><Canvas camera={{ position: [0, .25, 8.7], fov: 40 }} dpr={[1, 1.5]} frameloop={!visible ? 'never' : props.moving ? 'always' : 'demand'} gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }} onCreated={({ gl }) => { gl.domElement.addEventListener('webglcontextlost', () => setLost(true), { once: true }); }}>
      <Studio reset={props.reset} moving={props.moving}/><Cortex {...props}/>
    </Canvas></SceneBoundary>}
  </div>;
}
