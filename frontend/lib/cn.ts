import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function skillLabel(skillId: string): string {
  return skillId.replace(/_/g, " ");
}
