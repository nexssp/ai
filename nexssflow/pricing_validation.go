package nexssflow

import (
	"strings"

	"github.com/nexssp/cost"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

func validatePricingFields(actionName, model, rateCardID string, expected *cost.Currency, required bool) error {
	hasCard := rateCardID != ""
	hasCurrency := expected != nil
	if !hasCard && !hasCurrency && !required {
		return nil
	}
	if !hasCard || !hasCurrency || strings.TrimSpace(rateCardID) != rateCardID || strings.TrimSpace(model) == "" || strings.TrimSpace(model) != model {
		return xerr.Validation(actionName + ": model, rate_card_id, and expected_currency must be supplied as exact identifiers")
	}
	if !validCurrency(*expected) {
		return xerr.Validation(actionName + ": expected_currency must be a three-letter uppercase code")
	}
	return nil
}

func validateConfiguredPricing(actionName, model, rateCardID string, expected *cost.Currency) error {
	if err := validatePricingFields(actionName, model, rateCardID, expected, true); err != nil {
		return err
	}
	card, ok := llm.GlobalCatalog.GetRateCard(model, rateCardID)
	if !ok {
		return xerr.Validation(actionName + ": exact rate_card_id is not configured for the selected model")
	}
	if card.Currency == (cost.Currency{}) || card.Currency != *expected {
		return xerr.Validation(actionName + ": expected_currency does not match the selected rate card")
	}
	return nil
}

func validateTokenBasis(actionName string, promptTokens, cachedPromptTokens, completionTokens int, requirePrompt bool) error {
	if promptTokens < 0 || cachedPromptTokens < 0 || completionTokens < 0 || cachedPromptTokens > promptTokens || (requirePrompt && promptTokens == 0) {
		return xerr.Validation(actionName + ": token counts must be non-negative, prompt_tokens must be positive when priced, and cached input cannot exceed prompt input")
	}
	return nil
}

func validCurrency(currency cost.Currency) bool {
	for _, value := range currency {
		if value < 'A' || value > 'Z' {
			return false
		}
	}
	return true
}

func cloneCurrency(currency *cost.Currency) *cost.Currency {
	if currency == nil {
		return nil
	}
	cloned := *currency
	return &cloned
}
