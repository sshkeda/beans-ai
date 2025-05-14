"use client";

import { Message } from "@ai-sdk/react";
import { useEffect, useRef, useState } from "react";

export default function Chatbox({ messages }: { messages: Message[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    function handleScroll() {
      if (!containerRef.current) return;

      const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
      const isAtBottom = Math.abs(scrollHeight - clientHeight - scrollTop) < 10;

      if (isAtBottom) {
        setAutoScroll(true);
      } else if (autoScroll) {
        setAutoScroll(false);
      }
    }

    const container = containerRef.current;
    if (container) {
      container.addEventListener("scroll", handleScroll);
      return () => container.removeEventListener("scroll", handleScroll);
    }
  }, [autoScroll]);

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, autoScroll]);

  return (
    <div
      ref={containerRef}
      className="w-full bg-muted/40 aspect-square overflow-y-auto h-full p-2"
    >
      <p className="text-muted-foreground text-sm whitespace-pre-wrap">
        {messages.filter((m) => m.role === "assistant").map((m) => m.content)}
      </p>
    </div>
  );
}
