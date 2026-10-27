import { Fragment, type ElementType } from "react";

/**
 * Heading whose words can be staggered by GSAP. The accessible name is the unsplit
 * text (sr-only); the split words are decorative and fully visible without JS.
 */
export function SplitHeading({ as: Tag = "h2", text, className, id }: { as?: ElementType; text: string; className?: string; id?: string }) {
  return (
    <Tag className={className} id={id} data-split>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">
        {text.split(" ").map((w, i) => (
          <Fragment key={i}>{i > 0 && " "}<span className="w"><span className="wi">{w}</span></span></Fragment>
        ))}
      </span>
    </Tag>
  );
}
