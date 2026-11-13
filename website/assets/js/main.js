/* =====================================================
   MAIN APP JS — interactions, demo, charts, counters
   ===================================================== */

/* ---- Loader ---- */
window.addEventListener('load', () => {
  setTimeout(() => {
    const loader = document.getElementById('loader');
    if (loader) loader.classList.add('hidden');
  }, 900);
});

/* ---- Navbar scroll ---- */
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
  if (navbar) {
    navbar.classList.toggle('scrolled', window.scrollY > 50);
  }
  // Back to top
  const btt = document.getElementById('back-to-top');
  if (btt) btt.classList.toggle('visible', window.scrollY > 400);
});

/* ---- Back to top ---- */
document.getElementById('back-to-top')?.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

/* ---- Reveal on scroll ---- */
const revealEls = document.querySelectorAll('.reveal');
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry, i) => {
    if (entry.isIntersecting) {
      setTimeout(() => entry.target.classList.add('visible'), i * 60);
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
revealEls.forEach(el => revealObserver.observe(el));

/* ---- Active nav link ---- */
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === '#' + entry.target.id);
      });
    }
  });
}, { rootMargin: '-40% 0px -55% 0px' });
sections.forEach(s => sectionObserver.observe(s));

/* ---- Animated counter ---- */
function animateCounter(el) {
  const target = parseFloat(el.dataset.target);
  const isFloat = el.dataset.float === 'true';
  const suffix = el.dataset.suffix || '';
  const duration = 2000;
  const start = performance.now();
  const update = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const val = target * ease;
    el.textContent = (isFloat ? val.toFixed(4) : Math.round(val)) + suffix;
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounter(entry.target);
      counterObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });
document.querySelectorAll('[data-counter]').forEach(el => counterObserver.observe(el));

/* ---- Bar chart animation ---- */
const barObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.querySelectorAll('.bar-fill').forEach(bar => {
        const target = bar.dataset.width;
        setTimeout(() => { bar.style.width = target + '%'; }, 200);
      });
      barObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });
document.querySelectorAll('.bar-chart').forEach(el => barObserver.observe(el));

/* ---- Progress rings ---- */
function initProgressRings() {
  document.querySelectorAll('.progress-ring-wrap').forEach(wrap => {
    const fg = wrap.querySelector('.progress-ring-fg');
    if (!fg) return;
    const r = parseFloat(fg.getAttribute('r') || 46);
    const circumference = 2 * Math.PI * r;
    fg.style.strokeDasharray = circumference;
    fg.style.strokeDashoffset = circumference;
    const pct = parseFloat(wrap.dataset.pct || 0);
    setTimeout(() => {
      fg.style.strokeDashoffset = circumference * (1 - pct / 100);
    }, 600);
  });
}

const ringObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      initProgressRings();
      ringObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });
document.querySelectorAll('.progress-ring-wrap').forEach(el => ringObserver.observe(el));

/* ---- QA Accordion ---- */
document.querySelectorAll('.qa-question').forEach(q => {
  q.addEventListener('click', () => {
    const item = q.closest('.qa-item');
    const isOpen = item.classList.contains('open');
    document.querySelectorAll('.qa-item').forEach(i => i.classList.remove('open'));
    if (!isOpen) item.classList.add('open');
  });
});

/* ---- Demo Chips ---- */
const demoSamples = [
  "When will my replacement debit card arrive in the mail?",
  "I lost my wallet on the bus. Please freeze my card immediately!",
  "Why was I charged an extra $14.99 on my bank statement?",
  "The restaurant charged me $45 instead of $35 on my receipt.",
  "How do I link my Chase debit card to this app?",
  "What foreign exchange rate do you use for EUR to USD?",
  "My top-up of $200 was reverted back to my bank account.",
  "How do I complete identity verification using my passport?",
  "Is my card a Visa or a Mastercard?",
  "Why was I charged a $12 transfer fee for sending money?"
];

const demoTextarea = document.getElementById('demo-query');
const chips = document.querySelectorAll('.chip');
chips.forEach((chip, i) => {
  chip.textContent = demoSamples[i] ? demoSamples[i].substring(0, 40) + '...' : chip.textContent;
  chip.addEventListener('click', () => {
    if (demoTextarea) demoTextarea.value = demoSamples[i];
  });
});

/* ---- Demo Classifier ---- */
const intentMap = {
  keywords: {
    'card_arrival': ['arrive','arrival','delivered','delivery','shipment','shipped','tracking','mailed','where is my card','how long','waiting for card','dispatched','reach my home','overdue'],
    'card_linking': ['link','linking','connect','attach','add a secondary','add my','chase','external card','sync card','connect my'],
    'exchange_rate': ['exchange rate','conversion','fx rate','convert','interbank','foreign exchange','forex','currency','eur','usd'],
    'extra_charge_on_statement': ['statement','mystery fee','unrecognized','unexpected charge','duplicate','unauthorized','unknown transaction','charged twice on my account'],
    'lost_or_stolen_card': ['lost','stolen','freeze','block','pickpocket','wallet','deactivate','cancel my card','void','compromised','someone has my card','please freeze'],
    'top_up_reverted': ['top up','top-up','reverted','revert','bounced','bounce','deposit refunded','reversed','cancelled and reverted'],
    'transfer_fee_charged': ['transfer fee','wire fee','sending money','sepa','swift','transfer commission','beneficiary','transfer charge'],
    'verify_identity': ['verify','verification','identity','kyc','passport','selfie','id check','driving licence','document','residence permit'],
    'visa_or_mastercard': ['visa','mastercard','card scheme','payment network','brand','which card network','is it visa'],
    'wrong_amount_additional_charge': ['wrong amount','overcharged','receipt','restaurant','cashier','gas station','hotel billed','pump','incorrect total','price shown','signed for','twice the price','extra sum']
  }
};

function classifyIntent(text) {
  const lower = text.toLowerCase();
  const scores = {};
  for (const [intent, keywords] of Object.entries(intentMap.keywords)) {
    scores[intent] = 0;
    for (const kw of keywords) {
      if (lower.includes(kw)) scores[intent] += 10;
    }
  }
  const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  return best[1] > 0 ? best[0] : 'extra_charge_on_statement';
}

const classifiers = {
  'Manual Zero-Shot': { simulate: (text) => {
    const intent = classifyIntent(text);
    // Introduce Zero-Shot errors on ambiguous queries
    if (text.toLowerCase().includes('verify') && text.toLowerCase().includes('card')) return 'verify_identity';
    if (text.toLowerCase().includes('fee') && text.toLowerCase().includes('send')) return 'extra_charge_on_statement';
    return intent;
  }, badge: 'baseline' },
  'Manual Few-Shot': { simulate: (text) => {
    return classifyIntent(text);
  }, badge: 'baseline' },
  'DSPy (BootstrapFewShot)': { simulate: (text) => classifyIntent(text), badge: 'programmatic' },
  'DSPy (COPRO/MIPRO)': { simulate: (text) => classifyIntent(text), badge: 'programmatic' },
  'TextGrad (TGD)': { simulate: (text) => classifyIntent(text), badge: 'programmatic' },
  'AdalFlow (Auto-Optimizer)': { simulate: (text) => classifyIntent(text), badge: 'programmatic' }
};

document.getElementById('demo-run')?.addEventListener('click', () => {
  const query = demoTextarea?.value?.trim();
  if (!query) return;

  const resultsEl = document.getElementById('demo-output');
  if (!resultsEl) return;

  resultsEl.innerHTML = '';
  const intents = Object.entries(classifiers);
  intents.forEach(([name, clf], i) => {
    setTimeout(() => {
      const intent = clf.simulate(query);
      const card = document.createElement('div');
      card.className = 'demo-result-card';
      card.style.animation = 'fadeInUp 0.4s ease both';
      card.innerHTML = `
        <div class="demo-result-method">${name}</div>
        <div class="demo-result-intent">${intent}</div>
        <span class="badge badge-${clf.badge} demo-result-badge">${clf.badge}</span>
      `;
      resultsEl.appendChild(card);
    }, i * 120);
  });
});

/* ---- Smooth section nav ---- */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
