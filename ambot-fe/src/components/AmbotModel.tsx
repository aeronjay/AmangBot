import { useRef, useEffect, useState } from 'react'
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { Group } from 'three'

export function Model(props: any) {
  const group = useRef<Group>(null)
  const { nodes, materials, animations } = useGLTF('/ambotv1.1.glb') as any
  const { actions } = useAnimations(animations, group)
  const [index, setIndex] = useState(0)

  useEffect(() => {
    if (actions) {
        const actionNames = Object.keys(actions);
        if (actionNames.length > 0) {
            const actionName = actionNames[index % actionNames.length];
            const action = actions[actionName];
            action?.reset().fadeIn(0.5).play();
            
            return () => {
                action?.fadeOut(0.5);
            }
        }
    }
  }, [actions, index]);

  const handleClick = () => {
      if (actions) {
          const count = Object.keys(actions).length;
          if (count > 0) {
              setIndex((prev) => (prev + 1) % count);
          }
      }
  }

  return (
    <group ref={group} {...props} dispose={null} onClick={handleClick}>
      <group name="Scene">
        <group name="Armature">
          <skinnedMesh
            name="tripo_node_2d45e372-1007-4618-a246-081344c2f060"
            geometry={nodes['tripo_node_2d45e372-1007-4618-a246-081344c2f060'].geometry}
            material={materials['tripo_mat_2d45e372-1007-4618-a246-081344c2f060']}
            skeleton={nodes['tripo_node_2d45e372-1007-4618-a246-081344c2f060'].skeleton}
          />
          <primitive object={nodes.Root} />
        </group>
      </group>
    </group>
  )
}

useGLTF.preload('/ambotv1.1.glb')

export default function AmbotScene() {
  return (
    <div className="w-full h-full min-h-[400px]">
      <Canvas camera={{ position: [0, 0.5, 2.5], fov: 50 }}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={1.5} />
        <directionalLight position={[-5, 5, 5]} intensity={1} />
        <directionalLight position={[0, 2, 5]} intensity={1} />
        <Model position={[0.05, -.95, .1]} rotation={[0, -1.6, .3]} scale={2.0} />
        <OrbitControls 
            enableZoom={false} 
            enableRotate={false} 
            enablePan={false} 
        />
      </Canvas>
    </div>
  )
}
