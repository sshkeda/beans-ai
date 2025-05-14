"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectValue,
  SelectTrigger,
} from "@/components/ui/select";
import { Model, MODELS } from "@/lib/utils";
import type { Dispatch, SetStateAction } from "react";

export default function ModelSelector({
  model,
  setModel,
}: {
  model: Model;
  setModel: Dispatch<SetStateAction<Model>>;
}) {
  return (
    <Select
      defaultValue={model}
      onValueChange={(newValue) => setModel(newValue as Model)}
    >
      <SelectTrigger>
        <SelectValue placeholder="Select a model" />
      </SelectTrigger>
      <SelectContent>
        {MODELS.map((model) => (
          <SelectItem key={model} value={model}>
            {model}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
