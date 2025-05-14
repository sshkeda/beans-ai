"use client";

import { useEffect, useRef, useState } from "react";
import { Chess, Square } from "chess.js";
import { type Message, useChat } from "@ai-sdk/react";
import { toast } from "sonner";
import type { Model } from "@/lib/utils";

export function useBeans() {
  const [pgn, setPgn] = useState("");
  const [model, setModel] = useState<Model>("beans001-1.5B");
  const gameRef = useRef(new Chess());
  const initialMessages: Message[] = [
    {
      id: "1",
      role: "assistant",
      content:
        "\n\n\n\n\n\n\n\nPlay a game of chess against Beans,\na 1.5B parameter LLM based on DeepSeek R1.",
    },
  ];
  const messagesRef = useRef<Message[]>(initialMessages);

  const { messages, setMessages, setInput, append, status, stop } = useChat({
    experimental_throttle: 50,
    onError: () => toast.error("Beans is currently unavailable."),
    initialMessages,
  });

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  function makeMove(move: Parameters<Chess["move"]>[0] | string) {
    try {
      const result = gameRef.current.move(move);
      setPgn(gameRef.current.pgn());
      return result;
    } catch (error) {
      console.log(error);
      return null;
    }
  }

  function retry() {
    startBeansMove();
  }

  function reset() {
    stop();
    setMessages([]);
    setInput(""); // todo: for later chat support
    gameRef.current = new Chess();
    setPgn("");
  }

  async function startBeansMove() {
    setMessages([]);

    // I'm pretty sure there is a bug where append ⬇️ never returns a string.
    await append({
      content: "",
      role: "user",
      data: {
        model,
        fen: gameRef.current.fen(),
      },
    });

    await new Promise((resolve) => setTimeout(resolve, 1000));

    const response = messagesRef.current.at(-1)?.content;

    console.log(response);

    if (typeof response !== "string") return;

    const answer = response.match(/<answer>([\s\S]*)<\/answer>/)?.[1].trim();
    console.log(answer);
    if (!answer) return;

    makeMove(answer);
  }

  function onPieceDrop(sourceSquare: Square, targetSquare: Square) {
    if (status === "streaming") return false;
    if (gameRef.current.turn() === "b") return false;

    const move = makeMove({
      from: sourceSquare,
      to: targetSquare,
      promotion: "q", // todo: custom promotion
    });

    if (move) startBeansMove();

    return move !== null;
  }

  return {
    pgn,
    game: gameRef.current,
    setPgn,
    makeMove,
    onPieceDrop,
    reset,
    messages,
    setMessages,
    status,
    append,
    model,
    setModel,
    retry,
    stop,
  };
}
