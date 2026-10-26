import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { Checkpoint, Dataset, Document, Job, Page } from "./types";

export type Action = (
  work: () => Promise<unknown>,
  success?: string,
) => Promise<void>;
export type StartStage = (
  kind: string,
  config?: Record<string, unknown>,
  scope?: {
    document_ids?: string[];
    page_ids?: string[];
    mode?: string;
  },
) => Promise<void>;

export function useDocuments(projectId: string) {
  return useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => api<Document[]>(`/projects/${projectId}/documents`),
    enabled: !!projectId,
    refetchInterval: 3000,
  });
}
export function usePages(projectId: string, documentId: string) {
  return useQuery({
    queryKey: ["pages", projectId, documentId],
    queryFn: () =>
      api<Page[]>(`/projects/${projectId}/documents/${documentId}/pages`),
    enabled: !!projectId && !!documentId,
    refetchInterval: 4000,
  });
}
export function useDatasets(projectId: string) {
  return useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api<Dataset[]>(`/projects/${projectId}/datasets`),
    enabled: !!projectId,
  });
}
export function useCheckpoints(projectId: string) {
  return useQuery({
    queryKey: ["checkpoints", projectId],
    queryFn: () => api<Checkpoint[]>(`/projects/${projectId}/checkpoints`),
    enabled: !!projectId,
    refetchInterval: 5000,
  });
}
export function useJobs(projectId: string) {
  return useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api<Job[]>(`/projects/${projectId}/jobs`),
    enabled: !!projectId,
    refetchInterval: 2000,
  });
}
