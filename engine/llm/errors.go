package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/nexssp/kernel/xerr"
)

// FromHTTPStatus maps an upstream HTTP response into the kernel error
// taxonomy. Only documented transient statuses or exact structured error
// codes are retryable; arbitrary body text and malformed responses are not.
func FromHTTPStatus(status int, body []byte, providerName string) error {
	msg := fmt.Sprintf("%s: HTTP %d: %s", providerName, status, string(body))
	code := upstreamErrorCode(body)

	switch status {
	case http.StatusOK, http.StatusCreated, http.StatusAccepted:
		return nil
	case http.StatusBadRequest:
		return xerr.BadRequest(msg)
	case http.StatusUnauthorized:
		return xerr.Unauthorized(msg)
	case http.StatusForbidden:
		return xerr.Forbidden(msg)
	case http.StatusPaymentRequired:
		return xerr.Forbidden(msg)
	case http.StatusNotFound:
		return xerr.NotFound(msg)
	case http.StatusConflict:
		return xerr.Conflict(msg)
	case http.StatusTooManyRequests:
		switch code {
		case "", "rate_limit_exceeded", "too_many_requests", "rate_limited":
			return xerr.TooManyRequests(msg)
		case "insufficient_quota", "quota_exceeded", "billing_not_active", "billing_hard_limit_reached", "credit_balance_too_low", "account_deactivated":
			return xerr.Forbidden(msg)
		default:
			return xerr.Internal(msg)
		}
	case http.StatusRequestTimeout:
		return xerr.Timeout(msg)
	case http.StatusGatewayTimeout:
		return xerr.Timeout(msg)
	case http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable:
		return xerr.Unavailable(msg)
	default:
		return xerr.Internal(msg)
	}
}

// FromStreamHTTPStatus applies the provider classifier to stream startup
// failures. A successful but non-200 response is still a terminal protocol
// defect for a streaming endpoint, not a successful empty stream.
func FromStreamHTTPStatus(status int, body []byte, providerName string) error {
	if status >= 200 && status < 300 {
		return xerr.Internal(fmt.Sprintf("%s: stream endpoint returned unexpected successful status %d", providerName, status))
	}
	return FromHTTPStatus(status, body, providerName)
}

type upstreamErrorEnvelope struct {
	Code  string `json:"code"`
	Type  string `json:"type"`
	Error struct {
		Code string `json:"code"`
		Type string `json:"type"`
	} `json:"error"`
}

func upstreamErrorCode(body []byte) string {
	var payload upstreamErrorEnvelope
	if err := json.Unmarshal(body, &payload); err != nil {
		return ""
	}
	for _, value := range []string{payload.Error.Code, payload.Error.Type, payload.Code, payload.Type} {
		if value = strings.ToLower(strings.TrimSpace(value)); value != "" {
			return value
		}
	}
	return ""
}

func withErrorContext(message string, err error) error {
	if err == nil {
		return nil
	}
	appError := *xerr.From(err)
	appError.Message = message + ": " + appError.Message
	appError.Cause = err
	return &appError
}
