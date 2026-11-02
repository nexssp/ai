package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDecisionProviderPresetsDoNotChooseModelIDs(t *testing.T) {
	for _, name := range []string{providerXAI, providerInception} {
		if !IsKnownProvider(name) {
			t.Errorf("provider %q is not in the public registry", name)
		}
		provider, err := NewProvider(name, Config{})
		if err != nil {
			t.Fatalf("NewProvider(%q): %v", name, err)
		}
		if provider.Name() != name {
			t.Errorf("provider name = %q, want %q", provider.Name(), name)
		}
		switch p := provider.(type) {
		case *OpenAIProvider:
			if p.cfg.DefaultModel != "" {
				t.Errorf("%s preset selected model %q", name, p.cfg.DefaultModel)
			}
		case *InceptionProvider:
			if p.cfg.DefaultModel != "" {
				t.Errorf("%s preset selected model %q", name, p.cfg.DefaultModel)
			}
			if _, ok := provider.(DecisionProvider); !ok {
				t.Errorf("%s provider does not expose DecisionProvider", name)
			}
		default:
			t.Fatalf("provider %q constructed unexpected type %T", name, provider)
		}
	}
}

func TestFetchModelsUsesLiveCatalogForDecisionProviders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			t.Errorf("request = %s %s, want GET /v1/models", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "catalog-model-a"}, {"id": "catalog-model-b"}}})
	}))
	defer server.Close()

	for _, provider := range []string{providerXAI, providerInception} {
		models, err := FetchModels(context.Background(), provider, "test-key", server.URL+"/v1")
		if err != nil {
			t.Fatalf("FetchModels(%s): %v", provider, err)
		}
		if len(models) != 2 || models[0] != "catalog-model-a" || models[1] != "catalog-model-b" {
			t.Fatalf("FetchModels(%s) = %v", provider, models)
		}
	}
}

func TestFetchModelsReturnsErrorInsteadOfFabricatedFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "catalog unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	models, err := FetchModels(context.Background(), providerXAI, "", server.URL+"/v1")
	if err == nil {
		t.Fatal("expected unavailable catalog error")
	}
	if len(models) != 0 {
		t.Fatalf("unavailable catalog returned fabricated models: %v", models)
	}
}
