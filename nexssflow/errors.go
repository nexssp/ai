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
	if appErr, ok := errors.AsType[*xerr.AppError](err); ok && appErr != nil {
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
	kind := value.Kind()
	if (kind == reflect.Chan || kind == reflect.Func || kind == reflect.Interface || kind == reflect.Map || kind == reflect.Pointer || kind == reflect.Slice || kind == reflect.UnsafePointer) && value.IsNil() {
		return xerr.Internal("ai: provider builder returned a nil provider")
	}
	return nil
}
