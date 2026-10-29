"use client";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Float, PointMaterial, Points } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useReducedMotion } from "framer-motion";

function Field() {
  const ref = useRef<THREE.Group>(null);
  const reducedMotion = useReducedMotion();
  const { size } = useThree();
  const particleCount = size.width < 720 ? 360 : 720;
  const positions = useMemo(() => {
    const a = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const radius = 1.05 + Math.random() * 0.8, angle = Math.random() * Math.PI * 2;
      a[i * 3] = Math.cos(angle) * radius; a[i * 3 + 1] = (Math.random() - .5) * 2.4; a[i * 3 + 2] = Math.sin(angle) * radius;
    }
    return a;
  }, [particleCount]);
  useFrame((_, delta) => { if (!reducedMotion && ref.current) ref.current.rotation.y += delta * .025; });
  return <group ref={ref}>
    <Points positions={positions} stride={3}>
      <PointMaterial transparent color="#80dcff" size={0.018} sizeAttenuation depthWrite={false} opacity={0.75} />
    </Points>
    <mesh><icosahedronGeometry args={[0.55, 2]} /><meshStandardMaterial color="#13254d" emissive="#254bba" emissiveIntensity={1.1} metalness={0.65} roughness={0.3} wireframe /></mesh>
    <mesh><icosahedronGeometry args={[0.48, 2]} /><meshStandardMaterial color="#142246" emissive="#4b73ff" emissiveIntensity={0.35} metalness={0.5} roughness={0.25} /></mesh>
    <mesh rotation={[Math.PI / 2.6, 0.2, 0]}><torusGeometry args={[0.9, 0.003, 8, 160]} /><meshBasicMaterial color="#7865ff" transparent opacity={0.55} /></mesh>
    <mesh rotation={[0.8, 0, 0.9]}><torusGeometry args={[1.08, 0.002, 8, 160]} /><meshBasicMaterial color="#44c9ff" transparent opacity={0.42} /></mesh>
    <Float speed={reducedMotion ? 0 : 0.6} rotationIntensity={reducedMotion ? 0 : 0.12} floatIntensity={reducedMotion ? 0 : 0.2}><pointLight color="#4469ff" intensity={9} distance={4} /></Float>
  </group>;
}
export default function NeuralScene() {
  return <div className="scene" aria-hidden="true"><Canvas fallback={<div className="scene-fallback"><div className="fallback-orb" /></div>} camera={{ position: [0, 0, 4.1], fov: 44 }} dpr={[1, 1.35]} gl={{ alpha: true, antialias: false }}><ambientLight intensity={0.65} /><Field /></Canvas></div>;
}
