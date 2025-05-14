"use client";

import { Chessboard } from "react-chessboard";
import { Button } from "@/components/ui/button";
import { Flag, LoaderCircle, X, RotateCcw } from "lucide-react";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogCancel,
  AlertDialogAction,
  AlertDialogFooter,
  AlertDialogTrigger,
  AlertDialogDescription,
} from "@/components/ui/alert-dialog";
import { useBeans } from "@/hooks/use-beans";
import { cn } from "@/lib/utils";
import Chatbox from "@/components/chat-box";
import ModelSelector from "@/components/model-selector";

export default function Home() {
  const {
    pgn,
    game,
    onPieceDrop,
    reset,
    messages,
    status,
    retry,
    stop,
    model,
    setModel,
  } = useBeans();

  const showRetry =
    (status === "error" || status === "ready") &&
    game.turn() === "b" &&
    pgn.length > 1;

  return (
    <main className="flex items-center max-w-screen-md mx-auto justify-center min-h-screen">
      <div className="grid grid-cols-1 sm:mt-0 mt-16 sm:px-0 px-4 gap-4 w-full sm:grid-cols-2">
        <div className="col-span-1">
          <div className="w-full aspect-square relative bg-muted/40 z-10">
            <Chessboard position={game.fen()} onPieceDrop={onPieceDrop} />
          </div>
        </div>
        <div className="col-span-1 relative">
          <div className="absolute sm:-top-12 not-sm:-bottom-6 right-0 flex items-center gap-2">
            {status === "streaming" && (
              <LoaderCircle className="animate-spin stroke-muted-foreground h-3 w-3" />
            )}
            <ModelSelector model={model} setModel={setModel} />
          </div>
          <Chatbox messages={messages} />
        </div>
        <div className="col-span-1 relative">
          <div className="sm:absolute pb-8">
            <Moves pgn={pgn} />
          </div>
        </div>
        <div className="col-span-1 order-first sm:order-last">
          <div className="flex justify-between">
            <ResignButton pgn={pgn} reset={reset} />
            {showRetry ? (
              <Button size="sm" onClick={retry}>
                Retry generation <RotateCcw />
              </Button>
            ) : status === "streaming" ? (
              <Button variant="ghost" size="sm" onClick={stop}>
                Cancel generation <X />
              </Button>
            ) : null}
          </div>
        </div>
      </div>
    </main>
  );
}

function ResignButton({ pgn, reset }: { pgn: string; reset: () => void }) {
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm" className={cn(!pgn && "invisible")}>
          Resign <Flag />
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Are you sure you want to resign?</AlertDialogTitle>
          <AlertDialogDescription>
            Beans will win the game.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={reset}>Resign</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function Moves({ pgn }: { pgn: string }) {
  if (!pgn) return null;
  const moves = pgn
    .split(/\d+\./) // Split by move numbers
    .filter(Boolean) // Remove empty entries
    .map((move) => move.trim());

  return moves.map((moveText, index) => {
    const [white, black] = moveText.split(/\s+/).filter(Boolean);
    return (
      <div className="flex gap-2 font-mono" key={index}>
        <span className="text-muted-foreground">{index + 1}.</span>
        <span>{white}</span>
        <span>{black}</span>
      </div>
    );
  });
}
