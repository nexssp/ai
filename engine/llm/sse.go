// SSE parsing skeleton. The four streaming providers (OpenAI-compatible,
// DeepSeek-via-OpenAI, Anthropic, Gemini) differ only in the JSON shape of
// a single `data:` frame. The reader loop, the [DONE] terminator, the
// context check, and the close-on-exit all live here.
//
// Ollama does not use this file — it streams newline-delimited JSON, not
// SSE. Its reader stays local to ollama.go.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"iter"
	"strings"
)

// SSEChunkDecoder turns the JSON payload of a single SSE `data:` frame into
// a text fragment. Providers differ only in this function.
//
// Return values:
//
//   - text: fragment to yield. Empty fragments are skipped by the skeleton.
//   - done: true means the stream is complete and the reader stops.
//   - err:  decoding failed for this frame; the skeleton drops the frame
//     and continues. A single malformed frame does not tear down a
//     healthy stream.
type SSEChunkDecoder func(data []byte) (text string, done bool, err error)

// StreamSSE reads an SSE body line by line and calls decode on every
// `data:` payload. The function owns the reader lifecycle: it closes body
// on exit, checks ctx on every iteration, and stops on [DONE] or on the
// first decoder that returns done=true.
func StreamSSE(ctx context.Context, body io.ReadCloser, decode SSEChunkDecoder) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		defer body.Close()

		reader := bufio.NewReader(body)

		for {
			if ctx.Err() != nil {
				yield("", ctx.Err())

				return
			}

			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err != io.EOF {
					yield("", err)
				}

				return
			}

			trimmed := bytes.TrimSpace(line)
			if len(trimmed) == 0 || trimmed[0] == ':' {
				continue
			}

			after, ok := bytes.CutPrefix(trimmed, []byte("data:"))
			if !ok {
				continue
			}

			after = bytes.TrimSpace(after)

			// OpenAI-compatible providers send this literal terminator.
			// Anthropic terminates via a typed message_stop event inside
			// its decoder. Gemini simply ends the stream.
			if bytes.Equal(after, []byte("[DONE]")) {
				return
			}

			text, done, derr := decode(after)
			if derr != nil {
				continue
			}

			if text != "" {
				if !yield(text, nil) {
					return
				}
			}

			if done {
				return
			}
		}
	}
}

// DecodeOpenAIChunk handles the OpenAI / DeepSeek / Groq / most
// OpenAI-compatible SSE delta shape.
func DecodeOpenAIChunk(data []byte) (text string, done bool, err error) {
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &chunk); err != nil {
		return "", false, err
	}

	if len(chunk.Choices) == 0 {
		return "", false, nil
	}

	return chunk.Choices[0].Delta.Content, false, nil
}

// DecodeAnthropicChunk handles content_block_delta and message_stop.
func DecodeAnthropicChunk(data []byte) (text string, done bool, err error) {
	var event struct {
		Type  string `json:"type"`
		Delta struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"delta"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return "", false, err
	}

	if event.Type == "message_stop" {
		return "", true, nil
	}

	if event.Type == "content_block_delta" && event.Delta.Text != "" {
		return event.Delta.Text, false, nil
	}

	return "", false, nil
}

// DecodeGeminiChunk handles the candidates/parts/text shape. Multiple parts
// in a single frame are concatenated; the caller sees one fragment per frame.
func DecodeGeminiChunk(data []byte) (text string, done bool, err error) {
	var chunk struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(data, &chunk); err != nil {
		return "", false, err
	}

	if len(chunk.Candidates) == 0 {
		return "", false, nil
	}

	var b strings.Builder
	for _, p := range chunk.Candidates[0].Content.Parts {
		b.WriteString(p.Text)
	}

	return b.String(), false, nil
}

// ParseOpenAISSEStream parses an OpenAI-compatible SSE body into a
// standard iter.Seq2 iterator.
func ParseOpenAISSEStream(ctx context.Context, body io.ReadCloser) iter.Seq2[string, error] {
	return StreamSSE(ctx, body, DecodeOpenAIChunk)
}
