package llm

import (
	"bytes"
	"fmt"
	"net/http"

	"github.com/nexssp/kernel/xerr"
)

// FromHTTPStatus maps an upstream HTTP response into the kernel error
// taxonomy. Body markers are inspected before falling through to the
// status-based default, because OpenAI-compatible providers wrap
// transient conditions inside 400 responses.
func FromHTTPStatus(status int, body []byte, providerName string) error {
	msg := fmt.Sprintf("%s: HTTP %d: %s", providerName, status, string(body))

	switch status {
	case http.StatusOK, http.StatusCreated, http.StatusAccepted:
		return nil
	case http.StatusBadRequest:
		// Some providers return 400 for conditions that are semantically
		// transient — the caller should try a different provider/model.
		if isTransientBody(body) {
			return xerr.Unavailable(msg)
		}

		return xerr.BadRequest(msg)
	case http.StatusUnauthorized:
		return xerr.Unauthorized(msg)
	case http.StatusForbidden:
		return xerr.Forbidden(msg)
	case http.StatusNotFound:
		return xerr.NotFound(msg)
	case http.StatusConflict:
		return xerr.Conflict(msg)
	case http.StatusTooManyRequests:
		return xerr.TooManyRequests(msg)
	case http.StatusRequestTimeout:
		return xerr.Timeout(msg)
	case http.StatusServiceUnavailable, http.StatusBadGateway, http.StatusGatewayTimeout:
		return xerr.Unavailable(msg)
	default:
		if status >= 500 {
			return xerr.Unavailable(msg)
		}

		return xerr.Internal(msg)
	}
}

// transientMarkers are body substrings that signal a transient condition
// inside an otherwise non-transient HTTP status. Quote-delimited to avoid
// matching inside unrelated JSON values.
var transientMarkers = [][]byte{
	[]byte(`"context_length_exceeded"`),
	[]byte(`"overloaded"`),
	[]byte(`"rate_limit_exceeded"`),
	[]byte(`"server_error"`),
	[]byte(`"temporarily_unavailable"`),
}

func isTransientBody(body []byte) bool {
	for _, m := range transientMarkers {
		if bytes.Contains(body, m) {
			return true
		}
	}

	return false
}
