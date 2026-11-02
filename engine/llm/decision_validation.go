package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/nexssp/cost"
	"github.com/nexssp/kernel/xerr"
)

type decisionWireAnswer struct {
	Type          DecisionQuestionType      `json:"type,omitempty"`
	Noul          *float64                  `json:"noul,omitempty"`
	Choice        string                    `json:"choice,omitempty"`
	Score         *float64                  `json:"score,omitempty"`
	Probabilities map[string]float64        `json:"probabilities,omitempty"`
	Confidence    *float64                  `json:"confidence,omitempty"`
	Legend        map[string]string         `json:"legend,omitempty"`
	Spans         map[string][]DecisionSpan `json:"spans,omitempty"`
	NotStated     map[string]float64        `json:"not_stated,omitempty"`
}

func validateDecisionRequest(req DecisionRequest, configuredModel, provider string) (model string, allowed map[string]struct{}, validationErr error) {
	if provider == "" {
		provider = "decision"
	}
	model = req.Model
	if model == "" {
		model = configuredModel
	}
	if strings.TrimSpace(model) == "" {
		return "", nil, decisionValidationError(provider, "decision model is required in the request or provider configuration")
	}
	if req.MaxOutputTokens <= 0 {
		return "", nil, decisionValidationError(provider, "max_output_tokens must be a positive integer")
	}
	if err := validateDecisionPricing(req, model, provider); err != nil {
		return "", nil, err
	}
	if !json.Valid(req.State) {
		return "", nil, decisionValidationError(provider, "state must be valid JSON")
	}
	if len(req.Questions) == 0 {
		return "", nil, decisionValidationError(provider, "at least one question is required")
	}

	allowed, validationErr = allowedDecisionIDs(req.AllowedIDs, provider)
	if validationErr != nil {
		return "", nil, validationErr
	}
	seen := make(map[string]struct{}, len(req.Questions))
	for i := range req.Questions {
		question := &req.Questions[i]
		if strings.TrimSpace(question.ID) == "" {
			return "", nil, decisionValidationError(provider, "question IDs are required")
		}
		if _, exists := seen[question.ID]; exists {
			return "", nil, decisionValidationError(provider, "question IDs must be unique")
		}
		seen[question.ID] = struct{}{}
		if question.Threshold != nil && !probability(*question.Threshold) {
			message := fmt.Sprintf("question %q threshold must be between 0 and 1", question.ID)
			return "", nil, decisionValidationError(provider, message)
		}
		if err := validateDecisionQuestion(question, req.State, allowed, provider); err != nil {
			return "", nil, err
		}
	}
	return model, allowed, nil
}

func decisionValidationError(provider, message string) error {
	return xerr.Validation(fmt.Sprintf("%s: %s", provider, message))
}

func validateDecisionPricing(req DecisionRequest, model, provider string) error {
	hasRateCard := req.RateCardID != ""
	hasCurrency := req.ExpectedCurrency != nil
	if hasRateCard != hasCurrency {
		return decisionValidationError(provider, "rate_card_id and expected_currency must be supplied together")
	}
	if !hasRateCard {
		return nil
	}
	if strings.TrimSpace(req.RateCardID) != req.RateCardID || !validDecisionCurrency(*req.ExpectedCurrency) {
		return decisionValidationError(provider, "rate_card_id and expected_currency must be exact identifiers")
	}
	if req.EstimatedPromptTokens <= 0 || req.EstimatedCachedPromptTokens < 0 || req.EstimatedCompletionTokens < 0 || req.EstimatedCachedPromptTokens > req.EstimatedPromptTokens {
		return decisionValidationError(provider, "priced decision requests require a positive prompt-token estimate and valid cached/completion counts")
	}
	card, ok := GlobalCatalog.GetRateCard(model, req.RateCardID)
	if !ok || card.Currency == (cost.Currency{}) {
		return decisionValidationError(provider, "exact rate_card_id is not configured for the selected model")
	}
	if card.Currency != *req.ExpectedCurrency {
		return decisionValidationError(provider, "expected_currency does not match the selected rate card")
	}
	return nil
}

func allowedDecisionIDs(ids []string, provider string) (map[string]struct{}, error) {
	allowed := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) == "" {
			return nil, decisionValidationError(provider, "allowed IDs must not be empty")
		}
		if _, exists := allowed[id]; exists {
			return nil, decisionValidationError(provider, "allowed IDs must be unique")
		}
		allowed[id] = struct{}{}
	}
	return allowed, nil
}

func validateDecisionQuestion(question *DecisionQuestion, state json.RawMessage, allowed map[string]struct{}, provider string) error {
	switch question.Type {
	case DecisionNoul:
		return validateNoulQuestion(question, provider)
	case DecisionChoice:
		return validateChoiceQuestion(question, allowed, provider)
	case DecisionScore:
		return validateScoreQuestion(question, provider)
	case DecisionExtract:
		return validateExtractQuestion(question, state, provider)
	default:
		return decisionValidationError(provider, fmt.Sprintf("unsupported question type %q", question.Type))
	}
}

func validateNoulQuestion(question *DecisionQuestion, provider string) error {
	if question.Instructions == "" && question.TrueDescription == "" && question.FalseDescription == "" {
		return decisionValidationError(provider, fmt.Sprintf("noul question %q requires instructions or true/false descriptions", question.ID))
	}
	return nil
}

func validateChoiceQuestion(question *DecisionQuestion, allowed map[string]struct{}, provider string) error {
	criteria, err := criteriaMap(question.Criteria)
	if err != nil || len(criteria) < 1 || len(criteria) > 255 {
		return decisionValidationError(provider, fmt.Sprintf("choice question %q needs between 1 and 255 uniquely named criteria", question.ID))
	}
	if len(allowed) == 0 {
		return nil
	}
	for _, criterion := range question.Criteria {
		if _, ok := allowed[criterion.ID]; !ok {
			return decisionValidationError(provider, fmt.Sprintf("choice ID %q is outside the allowlist", criterion.ID))
		}
	}
	return nil
}

func validateScoreQuestion(question *DecisionQuestion, provider string) error {
	if len(question.Criteria) < 1 || len(question.Criteria) > 10 {
		return decisionValidationError(provider, fmt.Sprintf("score question %q needs between 1 and 10 ordered levels", question.ID))
	}
	for _, criterion := range question.Criteria {
		if strings.TrimSpace(criterion.Description) == "" {
			return decisionValidationError(provider, fmt.Sprintf("score question %q has an empty level description", question.ID))
		}
	}
	return nil
}

func validateExtractQuestion(question *DecisionQuestion, state json.RawMessage, provider string) error {
	if len(question.Criteria) > 26 {
		return decisionValidationError(provider, fmt.Sprintf("extract question %q may have at most 26 fields", question.ID))
	}
	if len(question.Criteria) == 0 && question.Instructions == "" {
		return decisionValidationError(provider, fmt.Sprintf("extract question %q requires criteria or instructions", question.ID))
	}
	if len(question.Criteria) > 0 {
		if _, err := criteriaMap(question.Criteria); err != nil {
			return decisionValidationError(provider, fmt.Sprintf("extract question %q needs uniquely named fields", question.ID))
		}
	}
	if isJSONObject(state) {
		if question.Source == "" {
			return decisionValidationError(provider, fmt.Sprintf("extract question %q needs source when state is an object", question.ID))
		}
		if _, err := decisionSourceText(state, question.Source); err != nil {
			return decisionValidationError(provider, fmt.Sprintf("extract question %q source: %v", question.ID, err))
		}
		return nil
	}
	if _, ok := jsonString(state); !ok {
		return decisionValidationError(provider, fmt.Sprintf("extract question %q needs string state or object state with a string source", question.ID))
	}
	return nil
}

func validDecisionCurrency(currency cost.Currency) bool {
	for _, value := range currency {
		if value < 'A' || value > 'Z' {
			return false
		}
	}
	return true
}

func criteriaMap(criteria []DecisionCriterion) (map[string]string, error) {
	if len(criteria) == 0 {
		return nil, errors.New("criteria are empty")
	}
	out := make(map[string]string, len(criteria))
	for _, criterion := range criteria {
		if strings.TrimSpace(criterion.ID) == "" {
			return nil, errors.New("criterion ID is empty")
		}
		if _, exists := out[criterion.ID]; exists {
			return nil, fmt.Errorf("duplicate criterion ID %q", criterion.ID)
		}
		out[criterion.ID] = criterion.Description
	}
	return out, nil
}

func decodeDecisionAnswer(question *DecisionQuestion, raw decisionWireAnswer, allowed map[string]struct{}, state json.RawMessage, provider string) (DecisionAnswer, error) {
	if raw.Type != "" && raw.Type != question.Type {
		message := fmt.Sprintf("type %q does not match %q", raw.Type, question.Type)
		return invalidDecisionAnswer(provider, question.ID, message)
	}
	if raw.Confidence != nil && !probability(*raw.Confidence) {
		return invalidDecisionAnswer(provider, question.ID, "confidence must be between 0 and 1")
	}

	answer := DecisionAnswer{ID: question.ID, Type: question.Type, Confidence: raw.Confidence}
	switch question.Type {
	case DecisionNoul:
		return decodeNoulAnswer(question, raw, answer, provider)
	case DecisionChoice:
		return decodeChoiceAnswer(question, raw, answer, allowed, provider)
	case DecisionScore:
		return decodeScoreAnswer(question, raw, answer, provider)
	case DecisionExtract:
		return decodeExtractAnswer(question, raw, answer, state, provider)
	default:
		return invalidDecisionAnswer(provider, question.ID, fmt.Sprintf("unsupported type %q", question.Type))
	}
}

func invalidDecisionAnswer(provider, questionID, message string) (DecisionAnswer, error) {
	return DecisionAnswer{}, xerr.Internal(fmt.Sprintf("%s: invalid answer %q: %s", provider, questionID, message))
}

func decodeNoulAnswer(question *DecisionQuestion, raw decisionWireAnswer, answer DecisionAnswer, provider string) (DecisionAnswer, error) {
	if raw.Noul == nil || !probability(*raw.Noul) {
		return invalidDecisionAnswer(provider, question.ID, "noul must be a probability between 0 and 1")
	}
	answer.Noul = raw.Noul
	return answer, nil
}

func decodeChoiceAnswer(question *DecisionQuestion, raw decisionWireAnswer, answer DecisionAnswer, allowed map[string]struct{}, provider string) (DecisionAnswer, error) {
	criteria, err := criteriaMap(question.Criteria)
	if err != nil {
		return invalidDecisionAnswer(provider, question.ID, err.Error())
	}
	if _, ok := criteria[raw.Choice]; !ok {
		return invalidDecisionAnswer(provider, question.ID, fmt.Sprintf("choice %q is not a declared criterion", raw.Choice))
	}
	if len(allowed) > 0 {
		if _, ok := allowed[raw.Choice]; !ok {
			return invalidDecisionAnswer(provider, question.ID, fmt.Sprintf("choice %q is outside the allowlist", raw.Choice))
		}
	}
	if err := validateProbabilities(raw.Probabilities, criteria); err != nil {
		return invalidDecisionAnswer(provider, question.ID, err.Error())
	}
	answer.Choice = raw.Choice
	answer.Probabilities = probabilitiesInOrder(question.Criteria, raw.Probabilities)
	return answer, nil
}

func decodeScoreAnswer(question *DecisionQuestion, raw decisionWireAnswer, answer DecisionAnswer, provider string) (DecisionAnswer, error) {
	if raw.Score == nil || math.IsNaN(*raw.Score) || math.IsInf(*raw.Score, 0) || *raw.Score < 0 || *raw.Score > float64(len(question.Criteria)-1) {
		return invalidDecisionAnswer(provider, question.ID, "score is outside the declared level range")
	}
	valid := make(map[string]string, len(question.Criteria))
	for i := range question.Criteria {
		valid[strconv.Itoa(i)] = question.Criteria[i].Description
	}
	if err := validateProbabilities(raw.Probabilities, valid); err != nil {
		return invalidDecisionAnswer(provider, question.ID, err.Error())
	}
	answer.Score = raw.Score
	for i := range question.Criteria {
		id := strconv.Itoa(i)
		criterion := &question.Criteria[i]
		description := raw.Legend[id]
		if description == "" {
			description = criterion.Description
		}
		answer.Legend = append(answer.Legend, DecisionCriterion{ID: id, Description: description})
		if value, ok := raw.Probabilities[id]; ok {
			answer.Probabilities = append(answer.Probabilities, DecisionProbability{ID: id, Probability: value})
		}
	}
	return answer, nil
}

func decodeExtractAnswer(question *DecisionQuestion, raw decisionWireAnswer, answer DecisionAnswer, state json.RawMessage, provider string) (DecisionAnswer, error) {
	fields := question.Criteria
	if len(fields) == 0 {
		fields = []DecisionCriterion{{ID: question.ID}}
	}
	valid := make(map[string]string, len(fields))
	for _, field := range fields {
		valid[field.ID] = field.Description
	}
	if err := validateExtractSpans(question, raw.Spans, valid, state, provider); err != nil {
		return DecisionAnswer{}, err
	}
	if err := validateExtractNotStated(question, raw.NotStated, valid, provider); err != nil {
		return DecisionAnswer{}, err
	}
	for _, field := range fields {
		answer.Extractions = append(answer.Extractions, DecisionExtraction{
			Field:                field.ID,
			Spans:                raw.Spans[field.ID],
			NotStatedProbability: raw.NotStated[field.ID],
		})
	}
	return answer, nil
}

func validateExtractSpans(question *DecisionQuestion, spansByField map[string][]DecisionSpan, valid map[string]string, state json.RawMessage, provider string) error {
	for field, spans := range spansByField {
		if _, ok := valid[field]; !ok {
			_, err := invalidDecisionAnswer(provider, question.ID, fmt.Sprintf("extract returned undeclared field %q", field))
			return err
		}
		for _, span := range spans {
			if !probability(span.Confidence) {
				_, err := invalidDecisionAnswer(provider, question.ID, fmt.Sprintf("span confidence for %q must be between 0 and 1", field))
				return err
			}
			if err := validateDecisionSpan(state, question.Source, span); err != nil {
				_, answerErr := invalidDecisionAnswer(provider, question.ID, fmt.Sprintf("invalid span for %q: %v", field, err))
				return answerErr
			}
		}
	}
	return nil
}

func validateExtractNotStated(question *DecisionQuestion, notStated map[string]float64, valid map[string]string, provider string) error {
	for field, value := range notStated {
		if _, ok := valid[field]; !ok {
			_, err := invalidDecisionAnswer(provider, question.ID, fmt.Sprintf("not_stated contains undeclared field %q", field))
			return err
		}
		if !probability(value) {
			_, err := invalidDecisionAnswer(provider, question.ID, fmt.Sprintf("not_stated probability for %q must be between 0 and 1", field))
			return err
		}
	}
	return nil
}

func validateProbabilities(values map[string]float64, valid map[string]string) error {
	for id, value := range values {
		if _, ok := valid[id]; !ok {
			return fmt.Errorf("probability contains undeclared outcome %q", id)
		}
		if !probability(value) {
			return fmt.Errorf("probability for %q must be between 0 and 1", id)
		}
	}
	return nil
}

func probabilitiesInOrder(criteria []DecisionCriterion, values map[string]float64) []DecisionProbability {
	out := make([]DecisionProbability, 0, len(values))
	for _, criterion := range criteria {
		if value, ok := values[criterion.ID]; ok {
			out = append(out, DecisionProbability{ID: criterion.ID, Probability: value})
		}
	}
	return out
}

func probability(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}

func isJSONObject(state json.RawMessage) bool {
	return strings.HasPrefix(strings.TrimSpace(string(state)), "{")
}

func jsonString(state json.RawMessage) (string, bool) {
	var value string
	if err := json.Unmarshal(state, &value); err != nil {
		return "", false
	}
	return value, true
}

func decisionSourceText(state json.RawMessage, source string) (string, error) {
	if !isJSONObject(state) {
		if value, ok := jsonString(state); ok {
			return value, nil
		}
		return "", errors.New("state is not a string or object")
	}
	if source == "" {
		return "", errors.New("source field is required")
	}
	var values map[string]json.RawMessage
	if err := json.Unmarshal(state, &values); err != nil {
		return "", err
	}
	raw, ok := values[source]
	if !ok {
		return "", fmt.Errorf("field %q is missing", source)
	}
	var text string
	if err := json.Unmarshal(raw, &text); err != nil {
		return "", fmt.Errorf("field %q must be a string", source)
	}
	return text, nil
}

func validateDecisionSpan(state json.RawMessage, source string, span DecisionSpan) error {
	text, err := decisionSourceText(state, source)
	if err != nil {
		return err
	}
	runes := []rune(text)
	if span.Start < 0 || span.End < span.Start || span.End > len(runes) {
		return fmt.Errorf("offsets [%d,%d) exceed source length %d", span.Start, span.End, len(runes))
	}
	if string(runes[span.Start:span.End]) != span.Text {
		return errors.New("text does not match source offsets")
	}
	return nil
}
