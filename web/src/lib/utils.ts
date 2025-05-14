import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const MODELS = ["beans001-1.5B", "gpt-4.1-nano"] as const;
export type Model = (typeof MODELS)[number];
