const resultData = {
  qwen2: {
    name: 'Qwen2-VL-7B',
    benchmarks: [
      { name: 'Video-MME', note: 'Overall', base: 53.7, memory: 60.5 },
      { name: 'MLVU', note: 'Overall', base: 56.9, memory: 65.7 },
      { name: 'LongVideoBench', note: 'Validation', base: 53.5, memory: 58.4 }
    ]
  },
  qwen3: {
    name: 'Qwen3-VL-8B',
    benchmarks: [
      { name: 'Video-MME', note: 'Overall', base: 57.4, memory: 64.7 },
      { name: 'MLVU', note: 'Overall', base: 57.2, memory: 66.5 },
      { name: 'LongVideoBench', note: 'Validation', base: 56.3, memory: 60.1 }
    ]
  },
  minicpm: {
    name: 'MiniCPM-V-4.5',
    benchmarks: [
      { name: 'Video-MME', note: 'Overall', base: 59.9, memory: 67.2 },
      { name: 'MLVU', note: 'Overall', base: 57.0, memory: 69.4 },
      { name: 'LongVideoBench', note: 'Validation', base: 55.6, memory: 62.0 }
    ]
  }
};

const exampleData = {
  retrieval: {
    image: 'assets/figure7-retrieval.png',
    alt: 'Question-conditioned MemoryCard retrieval visualization.',
    pill: 'Retrieve → allocate → reorder',
    title: 'Question-conditioned visual evidence',
    text: 'The retriever ranks cards from the reusable memory bank, then organizes them into 4 high-resolution, 8 medium-resolution, and 32 low-resolution inputs. Answer-critical details receive more pixels without sacrificing broad temporal coverage.'
  },
  selfread: {
    image: 'assets/figure6-selfread.png',
    alt: 'Uniform sampling compared with session-aware self-read construction.',
    pill: 'Segment → summarize → select',
    title: 'Question-agnostic semantic memory construction',
    text: 'Self-reading first divides a long video into coherent event sessions, then selects representative moments inside each session. This preserves event structure more faithfully than fixed global sampling.'
  }
};

const benchmarkGrid = document.querySelector('#benchmark-grid');

function renderBenchmarks(key) {
  const data = resultData[key];
  benchmarkGrid.innerHTML = data.benchmarks.map(item => {
    const delta = (item.memory - item.base).toFixed(1);
    return `
      <article class="benchmark-card">
        <div class="benchmark-top">
          <div><h3>${item.name}</h3><small>${item.note} accuracy</small></div>
          <span class="delta-badge">+${delta}</span>
        </div>
        <div class="bar-pair">
          <div class="bar-row"><span>Base</span><div class="bar-track"><div class="bar-fill base" style="width:${item.base}%"></div></div><strong>${item.base.toFixed(1)}</strong></div>
          <div class="bar-row"><span>MemoryCard</span><div class="bar-track"><div class="bar-fill memory" style="width:${item.memory}%"></div></div><strong>${item.memory.toFixed(1)}</strong></div>
        </div>
      </article>`;
  }).join('');
}

function bindTabKeyboard(tabs, selectTab) {
  tabs.forEach((tab, index) => {
    tab.addEventListener('keydown', event => {
      let nextIndex;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') nextIndex = 0;
      if (event.key === 'End') nextIndex = tabs.length - 1;
      if (nextIndex === undefined) return;
      event.preventDefault();
      selectTab(tabs[nextIndex]);
      tabs[nextIndex].focus();
    });
  });
}

const resultTabs = [...document.querySelectorAll('.tab')];
function selectResultTab(tab) {
  resultTabs.forEach(item => {
    const selected = item === tab;
    item.classList.toggle('active', selected);
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  benchmarkGrid.setAttribute('aria-labelledby', tab.id);
  renderBenchmarks(tab.dataset.backbone);
}
resultTabs.forEach(tab => tab.addEventListener('click', () => selectResultTab(tab)));
bindTabKeyboard(resultTabs, selectResultTab);
selectResultTab(resultTabs[0]);

const menuButton = document.querySelector('.menu-button');
const navLinks = document.querySelector('.nav-links');
const mobileMenu = window.matchMedia('(max-width: 860px)');

function setMenuState(open, restoreFocus = false) {
  const isOpen = mobileMenu.matches && open;
  navLinks.classList.toggle('open', isOpen);
  menuButton.setAttribute('aria-expanded', String(isOpen));
  menuButton.setAttribute('aria-label', isOpen ? 'Close navigation' : 'Open navigation');
  document.body.classList.toggle('menu-open', isOpen);
  navLinks.toggleAttribute('inert', mobileMenu.matches && !isOpen);
  if (mobileMenu.matches) navLinks.setAttribute('aria-hidden', String(!isOpen));
  else navLinks.removeAttribute('aria-hidden');
  if (restoreFocus) menuButton.focus();
}

menuButton.addEventListener('click', () => setMenuState(menuButton.getAttribute('aria-expanded') !== 'true'));
navLinks.querySelectorAll('a').forEach(link => link.addEventListener('click', () => setMenuState(false)));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && lightbox?.open) {
    lightbox.close();
    return;
  }
  if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') setMenuState(false, true);
});
document.addEventListener('pointerdown', event => {
  if (menuButton.getAttribute('aria-expanded') === 'true' && !event.target.closest('.site-header')) setMenuState(false);
});
mobileMenu.addEventListener('change', () => setMenuState(false));
setMenuState(false);

const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

const sections = [...document.querySelectorAll('main section[id]')];
const navAnchors = [...document.querySelectorAll('.nav-links a[href^="#"]')];
const sectionObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    navAnchors.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${entry.target.id}`));
  });
}, { rootMargin: '-35% 0px -58% 0px' });
sections.forEach(section => sectionObserver.observe(section));

document.querySelectorAll('[data-scroll-target]').forEach(button => {
  button.addEventListener('click', () => document.querySelector(button.dataset.scrollTarget)?.scrollIntoView({ behavior: 'smooth' }));
});

const lightbox = document.querySelector('#lightbox');
const lightboxImage = lightbox.querySelector('img');
let lightboxOpener;
document.querySelectorAll('[data-lightbox]').forEach(button => {
  button.addEventListener('click', () => {
    const sourceImage = button.querySelector('img');
    lightboxImage.src = button.dataset.lightbox;
    lightboxImage.alt = sourceImage?.alt ? `Expanded view: ${sourceImage.alt}` : 'Expanded paper figure';
    lightbox.setAttribute('aria-label', lightboxImage.alt);
    lightboxOpener = button;
    lightbox.showModal();
  });
});
lightbox.querySelector('.lightbox-close').addEventListener('click', () => lightbox.close());
lightbox.addEventListener('cancel', event => {
  event.preventDefault();
  lightbox.close();
});
lightbox.addEventListener('close', () => lightboxOpener?.focus());
lightbox.addEventListener('click', event => {
  const rect = lightbox.getBoundingClientRect();
  const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
  if (!inside) lightbox.close();
});

const exampleTabs = [...document.querySelectorAll('.example-tab')];
const examplePanel = document.querySelector('#example-stage');
let exampleSwapTimeout;

function selectExampleTab(tab) {
  const data = exampleData[tab.dataset.example];
  exampleTabs.forEach(item => {
    const selected = item === tab;
    item.classList.toggle('active', selected);
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  examplePanel.setAttribute('aria-labelledby', tab.id);
  const image = document.querySelector('#example-image');
  const button = image.closest('[data-lightbox]');
  image.style.opacity = '0';
  image.style.transform = 'scale(.985)';
  clearTimeout(exampleSwapTimeout);
  exampleSwapTimeout = setTimeout(() => {
    image.src = data.image;
    image.alt = data.alt;
    button.dataset.lightbox = data.image;
    button.setAttribute('aria-label', `Open ${data.title.toLowerCase()} figure`);
    document.querySelector('#example-copy').innerHTML = `<span class="pill">${data.pill}</span><h3>${data.title}</h3><p>${data.text}</p>`;
    image.style.opacity = '1';
    image.style.transform = 'none';
  }, 160);
}

exampleTabs.forEach(tab => tab.addEventListener('click', () => selectExampleTab(tab)));
bindTabKeyboard(exampleTabs, selectExampleTab);

const toast = document.querySelector('#toast');
document.querySelector('#copy-bibtex').addEventListener('click', async () => {
  const text = document.querySelector('#bibtex-code').textContent;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement('textarea');
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 1800);
});
