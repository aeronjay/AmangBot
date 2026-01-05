import { useRef, useEffect, useState } from 'react'
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { Group } from 'three'

const ANIMATION_MAPPING: Record<string, number[]> = {
    idle: [2, 4, 5, 13, 16, 23],
    user_typing: [19, 20],
    answering: [10, 12],
    after_answering: [1, 6, 18, 24],
    complain: [9, 14, 21, 22],
    greeting: [15, 17],
    clicked: [0, 7]
};

export function Model({ botState, ...props }: any) {
  const group = useRef<Group>(null)
  const { scene, animations } = useGLTF('/ambotv2.glb') as any
  const { actions } = useAnimations(animations, group)
  const [currentActionName, setCurrentActionName] = useState<string | null>(null);
  const [localState, setLocalState] = useState<string | null>(null);

  useEffect(() => {
    if (actions) {
        const actionNames = Object.keys(actions);
        
        const stateToUse = localState || botState || 'idle';
        const availableIndices = ANIMATION_MAPPING[stateToUse] || ANIMATION_MAPPING['idle'];
        
        // Pick a random index
        const randomIndex = availableIndices[Math.floor(Math.random() * availableIndices.length)];
        
        if (randomIndex < actionNames.length) {
            const actionName = actionNames[randomIndex];
            
            if (currentActionName !== actionName) {
                // Fade out previous action
                if (currentActionName && actions[currentActionName]) {
                    actions[currentActionName]?.fadeOut(0.5);
                }

                // Play new action
                const action = actions[actionName];
                if (action) {
                    action.reset().fadeIn(0.5).play();
                    setCurrentActionName(actionName);
                }
            }
        }
    }
  }, [actions, botState, localState]);

  const handleClick = () => {
      setLocalState('clicked');
      setTimeout(() => {
          setLocalState(null);
      }, 4000);
  }

  return (
    <group ref={group} {...props} dispose={null} onClick={handleClick}>
      <primitive object={scene} />
    </group>
  )
}

useGLTF.preload('/ambotv2.glb')

export default function AmbotScene({ botState }: { botState?: string }) {
  return (
    <div className="w-full h-full min-h-[400px]">
      <Canvas camera={{ position: [0, 0.5, 2.5], fov: 50 }}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={1.5} />
        <directionalLight position={[-5, 5, 5]} intensity={1} />
        <directionalLight position={[0, 2, 5]} intensity={1} />
        <Model botState={botState} position={[0.05, -.95, .1]} rotation={[0, -1.6, .3]} scale={1.7} />
        <OrbitControls 
            enableZoom={false} 
            enableRotate={false} 
            enablePan={false} 
        />
      </Canvas>
    </div>
  )
}
