/* =====================================================
   PARTICLES — lightweight star/node background
   ===================================================== */
class ParticleSystem {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.particles = [];
    this.connections = [];
    this.mouse = { x: -1000, y: -1000 };
    this.resize();
    this.init();
    window.addEventListener('resize', () => this.resize());
    document.addEventListener('mousemove', e => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  init() {
    this.particles = [];
    const count = Math.min(80, Math.floor(window.innerWidth / 18));
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: Math.random() * 1.8 + 0.6,
        alpha: Math.random() * 0.5 + 0.2
      });
    }
  }

  draw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    // Draw connections
    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const dx = this.particles[i].x - this.particles[j].x;
        const dy = this.particles[i].y - this.particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 140) {
          this.ctx.beginPath();
          this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
          this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
          this.ctx.strokeStyle = `rgba(108,99,255,${0.12 * (1 - dist / 140)})`;
          this.ctx.lineWidth = 0.8;
          this.ctx.stroke();
        }
      }
    }
    // Draw particles
    for (const p of this.particles) {
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      this.ctx.fillStyle = `rgba(168,157,255,${p.alpha})`;
      this.ctx.fill();

      // Move
      p.x += p.vx;
      p.y += p.vy;

      // Repel from mouse
      const dx = p.x - this.mouse.x;
      const dy = p.y - this.mouse.y;
      const md = Math.sqrt(dx * dx + dy * dy);
      if (md < 100) {
        p.x += (dx / md) * 1.5;
        p.y += (dy / md) * 1.5;
      }

      // Bounce
      if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;
    }

    requestAnimationFrame(() => this.draw());
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('particles-canvas');
  if (canvas) {
    const ps = new ParticleSystem(canvas);
    ps.draw();
  }
});
