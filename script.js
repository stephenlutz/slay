import GLea from 'https://unpkg.com/glea?module';

const frag = document.getElementById('fragmentShader').textContent;
const vert = document.getElementById('vertexShader').textContent;

let texture = null;

const glea = new GLea({
  shaders: [
    GLea.fragmentShader(frag),
    GLea.vertexShader(vert)
  ],
  buffers: {
    'pos': GLea.buffer(2, [1, 1,  -1, 1,  1,-1,  -1,-1])
  }
}).create();

function loop(time) {
  const { gl } = glea;
  glea.clear();
  glea.uni('width', glea.width);
  glea.uni('height', glea.height);
  glea.uni('time', time * .0002);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  requestAnimationFrame(loop);
}

function updateCanvasSize() {
  // Force recalculation of canvas size
  const pixelRatio = window.devicePixelRatio || 1;
  glea.resize();
  
  // Ensure the canvas is properly sized for mobile
  document.documentElement.style.setProperty('--vh', `${window.innerHeight * 0.01}px`);
}

function setup() {
  const { gl } = glea;
  
  // Initial size setup
  updateCanvasSize();
  
  // Handle both orientation changes and resize events
  window.addEventListener('resize', updateCanvasSize);
  window.addEventListener('orientationchange', updateCanvasSize);
  
  // Handle iOS Safari specific issues
  document.addEventListener('touchmove', e => e.preventDefault(), { passive: false });
  
  // Address the iOS visual viewport issues
  window.visualViewport?.addEventListener('resize', updateCanvasSize);
  
  loop(0);
}

// Fix for Safari mobile full height
function resetHeight() {
  document.body.style.height = window.innerHeight + 'px';
}

window.addEventListener('load', () => {
  resetHeight();
  setup();
});

window.addEventListener('resize', resetHeight);
window.addEventListener('orientationchange', resetHeight);
