// Home-page motion only: GSAP intro + word reveals, Lenis smooth scroll wired to ScrollTrigger.
// Under prefers-reduced-motion nothing is initialised and the static layout is the final state.
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";

gsap.registerPlugin(ScrollTrigger);

export const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Smooth scroll + section heading reveals. Returns cleanup. */
export function initHomeScroll(root: HTMLElement): () => void {
  if (reducedMotion()) return () => {};
  const lenis = new Lenis({ lerp: 0.12 });
  const raf = (t: number) => lenis.raf(t * 1000);
  lenis.on("scroll", ScrollTrigger.update);
  gsap.ticker.add(raf);
  gsap.ticker.lagSmoothing(0);

  const ctx = gsap.context(() => {
    root.querySelectorAll<HTMLElement>("h2[data-split]").forEach((h) => {
      gsap.from(h.querySelectorAll(".wi"), {
        yPercent: 105, duration: 0.7, ease: "power3.out", stagger: 0.05,
        scrollTrigger: { trigger: h, start: "top 85%", once: true },
      });
    });
  }, root);
  document.fonts?.ready.then(() => ScrollTrigger.refresh());

  return () => {
    ctx.revert();
    gsap.ticker.remove(raf);
    lenis.destroy();
  };
}

/** Hero intro: headline words rise, supporting copy fades up alongside. Returns cleanup. */
export function heroIntro(root: HTMLElement): () => void {
  if (reducedMotion() || document.hidden) return () => {};
  let tl: gsap.core.Timeline | undefined;
  const ctx = gsap.context(() => {
    tl = gsap.timeline({ defaults: { ease: "power3.out" } })
      .from("h1 .wi", { yPercent: 110, duration: 0.8, stagger: 0.045 })
      .from("[data-intro]", { opacity: 0, y: 12, duration: 0.5, stagger: 0.08 }, 0.25);
  }, root);
  // rAF is paused in background tabs/headless captures: never leave copy hidden mid-intro.
  const done = setTimeout(() => tl?.progress(1), 2500);
  return () => { clearTimeout(done); ctx.revert(); };
}

/** Board rows settle in once, the first time data arrives. */
export function boardIntro(rows: Element[]) {
  if (reducedMotion() || !rows.length) return;
  gsap.from(rows, { opacity: 0, y: 10, duration: 0.5, ease: "power2.out", stagger: 0.035, clearProps: "all" });
}
