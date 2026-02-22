import { motion as Motion, useMotionTemplate, useMotionValue, useSpring } from 'framer-motion'

function AuroraBackground({ children }) {
  const pointerX = useMotionValue(50)
  const pointerY = useMotionValue(50)

  const smoothX = useSpring(pointerX, {
    stiffness: 70,
    damping: 20,
    mass: 0.45,
  })
  const smoothY = useSpring(pointerY, {
    stiffness: 70,
    damping: 20,
    mass: 0.45,
  })

  const cyanX = useMotionTemplate`${smoothX}%`
  const cyanY = useMotionTemplate`${smoothY}%`
  const whiteX = useMotionTemplate`calc(${smoothX}% + 12%)`
  const whiteY = useMotionTemplate`calc(${smoothY}% - 10%)`
  const indigoX = useMotionTemplate`calc(${smoothX}% - 14%)`
  const indigoY = useMotionTemplate`calc(${smoothY}% + 14%)`

  const cyanGradient = useMotionTemplate`radial-gradient(circle at ${cyanX} ${cyanY}, rgba(102, 252, 255, 0.76), transparent 56%)`
  const whiteGradient = useMotionTemplate`radial-gradient(circle at ${whiteX} ${whiteY}, rgba(242, 242, 243, 0.94), transparent 62%)`
  const indigoGradient = useMotionTemplate`radial-gradient(circle at ${indigoX} ${indigoY}, rgba(79, 64, 140, 0.58), transparent 60%)`

  const handleMouseMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 100
    const y = ((event.clientY - rect.top) / rect.height) * 100

    pointerX.set(Math.min(100, Math.max(0, x)))
    pointerY.set(Math.min(100, Math.max(0, y)))
  }

  const handleMouseLeave = () => {
    pointerX.set(50)
    pointerY.set(50)
  }

  return (
    <div className="aurora-page" onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave}>
      <Motion.div
        className="aurora-layer aurora-layer-cyan"
        style={{ backgroundImage: cyanGradient }}
        animate={{ scale: [1, 1.06, 1], x: ['-2%', '3%', '-1%'], y: ['-2%', '1%', '-2%'] }}
        transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
      />
      <Motion.div
        className="aurora-layer aurora-layer-white"
        style={{ backgroundImage: whiteGradient }}
        animate={{ scale: [1.02, 1.1, 1.02], x: ['2%', '-2%', '2%'], y: ['1%', '4%', '1%'] }}
        transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
      />
      <Motion.div
        className="aurora-layer aurora-layer-indigo"
        style={{ backgroundImage: indigoGradient }}
        animate={{ scale: [1, 1.08, 1], x: ['1%', '-3%', '1%'], y: ['3%', '-1%', '3%'] }}
        transition={{ duration: 16, repeat: Infinity, ease: 'easeInOut' }}
      />
      {children}
    </div>
  )
}

export default AuroraBackground
