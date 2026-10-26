import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";

const schema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    td: ["colSpan", "rowSpan"],
    th: ["colSpan", "rowSpan"],
    "*": ["lang"],
  },
};

export function Preview({ content }: { content: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, schema]]}
        disallowedElements={[
          "img",
          "script",
          "style",
          "iframe",
          "object",
          "link",
          "svg",
          "video",
          "audio",
        ]}
        urlTransform={() => ""}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
