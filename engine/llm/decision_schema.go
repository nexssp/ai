package llm

import "strconv"

const (
	schemaTypeObject = "object"
	schemaTypeString = "string"
	schemaTypeNumber = "number"
)

func decisionJSONSchema(req DecisionRequest) *JSONSchema {
	answers := make(map[string]JSONSchema, len(req.Questions))
	questionIDs := make([]string, 0, len(req.Questions))
	for i := range req.Questions {
		question := &req.Questions[i]
		questionIDs = append(questionIDs, question.ID)
		answers[question.ID] = decisionAnswerSchema(question)
	}
	return &JSONSchema{
		Type: schemaTypeObject,
		Properties: map[string]JSONSchema{
			"answers": objectSchema(answers, questionIDs),
		},
		Required:             []string{"answers"},
		AdditionalProperties: new(false),
	}
}

func decisionAnswerSchema(question *DecisionQuestion) JSONSchema {
	properties := map[string]JSONSchema{
		"type":       {Type: schemaTypeString, Enum: []string{string(question.Type)}},
		"confidence": {Type: schemaTypeNumber},
	}
	required := []string{"type"}

	switch question.Type {
	case DecisionNoul:
		properties["noul"] = JSONSchema{Type: schemaTypeNumber}
		required = append(required, "noul")
	case DecisionChoice:
		choiceIDs := make([]string, 0, len(question.Criteria))
		probabilityProperties := make(map[string]JSONSchema, len(question.Criteria))
		for _, criterion := range question.Criteria {
			choiceIDs = append(choiceIDs, criterion.ID)
			probabilityProperties[criterion.ID] = JSONSchema{Type: schemaTypeNumber}
		}
		properties["choice"] = JSONSchema{Type: schemaTypeString, Enum: choiceIDs}
		properties["probabilities"] = objectSchema(probabilityProperties, nil)
		required = append(required, "choice")
	case DecisionScore:
		probabilityProperties := make(map[string]JSONSchema, len(question.Criteria))
		legendProperties := make(map[string]JSONSchema, len(question.Criteria))
		for i := range question.Criteria {
			id := strconv.Itoa(i)
			probabilityProperties[id] = JSONSchema{Type: schemaTypeNumber}
			legendProperties[id] = JSONSchema{Type: schemaTypeString}
		}
		properties["score"] = JSONSchema{Type: schemaTypeNumber}
		properties["probabilities"] = objectSchema(probabilityProperties, nil)
		properties["legend"] = objectSchema(legendProperties, nil)
		required = append(required, "score")
	case DecisionExtract:
		fields := question.Criteria
		if len(fields) == 0 {
			fields = []DecisionCriterion{{ID: question.ID}}
		}
		span := JSONSchema{
			Type: schemaTypeObject,
			Properties: map[string]JSONSchema{
				"text":       {Type: schemaTypeString},
				"start":      {Type: "integer"},
				"end":        {Type: "integer"},
				"confidence": {Type: schemaTypeNumber},
			},
			Required:             []string{"text", "start", "end", "confidence"},
			AdditionalProperties: new(false),
		}
		spanList := JSONSchema{Type: "array", Items: &span}
		spanProperties := make(map[string]JSONSchema, len(fields))
		notStatedProperties := make(map[string]JSONSchema, len(fields))
		for _, field := range fields {
			spanProperties[field.ID] = spanList
			notStatedProperties[field.ID] = JSONSchema{Type: schemaTypeNumber}
		}
		properties["spans"] = objectSchema(spanProperties, nil)
		properties["not_stated"] = objectSchema(notStatedProperties, nil)
	}

	return JSONSchema{
		Type:                 schemaTypeObject,
		Properties:           properties,
		Required:             required,
		AdditionalProperties: new(false),
	}
}

func objectSchema(properties map[string]JSONSchema, required []string) JSONSchema {
	return JSONSchema{
		Type:                 schemaTypeObject,
		Properties:           properties,
		Required:             required,
		AdditionalProperties: new(false),
	}
}
