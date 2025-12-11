/* const navbar = document.getElementById('navbar');
let lastScrollTop = 0;
const MOBILE_BREAKPOINT = 768; // Matches Tailwind's 'md' breakpoint

window.addEventListener('scroll', function () {
  // 1. Efficiency Check:
  // If we are on mobile (where CSS hides the navbar), do nothing.
  if (window.innerWidth < MOBILE_BREAKPOINT) return;

  let scrollTop = window.pageY || document.documentElement.scrollTop;

  // Prevent negative scrolling bounce (common on macOS/iOS)
  if (scrollTop < 0) scrollTop = 0;

  // --- Shadow Logic ---
  if (scrollTop > 10) {
    navbar.classList.add('shadow-md');
  } else {
    navbar.classList.remove('shadow-md');
  }

  // --- Smart Scroll Logic ---
  const navbarHeight = navbar.offsetHeight;

  // If scrolling DOWN and past the navbar -> Slide Up (Hide)
  if (scrollTop > lastScrollTop && scrollTop > navbarHeight) {
    navbar.style.transform = `translateY(-${navbarHeight}px)`;
  }
  // If scrolling UP -> Slide Down (Show)
  else {
    navbar.style.transform = 'translateY(0)';
  }

  lastScrollTop = scrollTop;
}); */