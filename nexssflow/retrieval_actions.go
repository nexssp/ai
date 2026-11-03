package nexssflow

import (
	"context"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/retrieval"
)

func newRetrievalActions(global GlobalOptions, build providerBuilder, service *retrieval.Service) []action.AnyAction {
	return []action.AnyAction{
		newRetrievalIndexAction(service),
		newRetrievalDeleteAction(service),
		newRetrievalSearchAction(service),
		newRAGAnswerAction(global, build, service),
	}
}

func newRetrievalIndexAction(service *retrieval.Service) action.AnyAction {
	return action.New[retrieval.Document, retrieval.IndexResult]("ai.retrieval.index", func(ctx context.Context, req retrieval.Document) (retrieval.IndexResult, error) {
		if service == nil {
			return retrieval.IndexResult{}, xerr.Internal("ai.retrieval.index: retrieval service is not configured")
		}
		return service.Index(ctx, req)
	}).Description("Chunk and embed a document, atomically replacing its prior chunks in the namespace; requires an injected retrieval service").Tag("ai", "retrieval", "index").Build()
}

func newRetrievalDeleteAction(service *retrieval.Service) action.AnyAction {
	return action.New[RetrievalDeleteRequest, RetrievalDeleteResponse]("ai.retrieval.delete", func(ctx context.Context, req RetrievalDeleteRequest) (RetrievalDeleteResponse, error) {
		if service == nil {
			return RetrievalDeleteResponse{}, xerr.Internal("ai.retrieval.delete: retrieval service is not configured")
		}
		if err := service.Delete(ctx, req.Namespace, req.DocumentID); err != nil {
			return RetrievalDeleteResponse{}, err
		}
		return RetrievalDeleteResponse{Namespace: req.Namespace, DocumentID: req.DocumentID, Deleted: true}, nil
	}).Description("Delete every indexed chunk for one document within the given namespace").Tag("ai", "retrieval", "delete").Build()
}

func newRetrievalSearchAction(service *retrieval.Service) action.AnyAction {
	return action.New[retrieval.SearchRequest, retrieval.SearchResponse]("ai.retrieval.search", func(ctx context.Context, req retrieval.SearchRequest) (retrieval.SearchResponse, error) {
		if service == nil {
			return retrieval.SearchResponse{}, xerr.Internal("ai.retrieval.search: retrieval service is not configured")
		}
		return service.Search(ctx, req)
	}).Description("Embed a query and return ranked, namespace-scoped source chunks").Tag("ai", "retrieval", "search").Build()
}

func newRAGAnswerAction(global GlobalOptions, build providerBuilder, service *retrieval.Service) action.AnyAction {
	return action.New[RAGAnswerActionRequest, retrieval.AnswerResponse]("ai.rag.answer", func(ctx context.Context, req RAGAnswerActionRequest) (retrieval.AnswerResponse, error) {
		if service == nil {
			return retrieval.AnswerResponse{}, xerr.Internal("ai.rag.answer: retrieval service is not configured")
		}
		if build == nil {
			return retrieval.AnswerResponse{}, xerr.Internal("ai.rag.answer: provider builder is unavailable")
		}
		provider, err := build(global, LLMDecl{Provider: req.Provider, Model: req.Model, BaseURL: req.BaseURL})
		if err != nil {
			return retrieval.AnswerResponse{}, completionError(err)
		}
		if err := requireProvider(provider); err != nil {
			return retrieval.AnswerResponse{}, err
		}
		return service.Answer(ctx, provider, retrieval.AnswerRequest{
			Namespace:    req.Namespace,
			Query:        req.Query,
			TopK:         req.TopK,
			MinScore:     req.MinScore,
			Model:        req.Model,
			SystemPrompt: req.SystemPrompt,
			MaxTokens:    req.MaxTokens,
		})
	}).Description("Answer a question with retrieved source excerpts treated as untrusted data; returns the exact sources used, while citations remain model-generated").Tag("ai", "rag", "answer").Build()
}
