package nexssflow

import (
	"errors"
	"reflect"

	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/engine/llm"
)

// completionError preserves provider classifications and classifies raw errors.
// Unknown errors are internal defects, not evidence that a retry is safe.
func completionError(err error) error {
	if err == nil {
		return nil
	}
	var typed *xerr.AppError
	if errors.As(err, &typed) {
		return err
	}
	return xerr.From(err)
}

// errorContext adds action context without mutating a shared provider error.
// Copying the typed error retains its kind, validation details, and stack;
// keeping the original as the cause preserves errors.Is and errors.As.
func errorContext(message string, err error) error {
	if err == nil {
		return nil
	}
	out := *xerr.From(err)
	out.Message = message + ": " + out.Message
	out.Cause = err
	return &out
}

// requireProvider catches both nil interfaces and typed nil implementations.
func requireProvider(provider llm.Provider) error {
	if provider == nil {
		return xerr.Internal("ai: provider builder returned a nil provider")
	}
	value := reflect.ValueOf(provider)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		if value.IsNil() {
			return xerr.Internal("ai: provider builder returned a nil provider")
		}
	}
	return nil
}
