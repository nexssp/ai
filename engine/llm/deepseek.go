package llm

import (
	"context"
	"iter"
	"regexp"
	"strings"
)

var thinkTagRegex = regexp.MustCompile(`(?s)<think>(.*?)</think>`)

type DeepSeekProvider struct {
	underlying Provider
}

func NewDeepSeek(cfg Config) *DeepSeekProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.deepseek.com/v1"
	}

	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "deepseek-chat"
	}

	if cfg.Name == "" {
		cfg.Name = "deepseek"
	}

	return &DeepSeekProvider{underlying: NewOpenAI(cfg)}
}

func (p *DeepSeekProvider) Name() string { return p.underlying.Name() }

func (p *DeepSeekProvider) Complete(ctx context.Context, req Request) (Response, error) {
	res, err := p.underlying.Complete(ctx, req)
	if err != nil {
		return Response{}, err
	}

	if match := thinkTagRegex.FindStringSubmatch(res.Content); len(match) > 1 {
		res.ReasoningContent = strings.TrimSpace(match[1])

		stripped := strings.TrimSpace(thinkTagRegex.ReplaceAllString(res.Content, ""))
		if stripped != "" || req.IncludeThinking {
			res.Content = stripped
		}
	}

	return res, nil
}

func (p *DeepSeekProvider) Stream(ctx context.Context, req Request) (iter.Seq2[string, error], error) {
	rawStream, err := p.underlying.Stream(ctx, req)
	if err != nil {
		return nil, err
	}

	if req.IncludeThinking {
		return rawStream, nil
	}

	return func(yield func(string, error) bool) {
		insideThink := false

		var buf strings.Builder

		for chunk, streamErr := range rawStream {
			if streamErr != nil {
				yield("", streamErr)

				return
			}

			buf.WriteString(chunk)
			curr := buf.String()

			for len(curr) > 0 {
				if !insideThink {
					openIdx := strings.Index(curr, "<think>")
					if openIdx == -1 {
						hasPartial := false

						for i := 1; i < len("<think>"); i++ {
							if strings.HasSuffix(curr, "<think>"[:i]) {
								safeLen := len(curr) - i
								if safeLen > 0 {
									if !yield(curr[:safeLen], nil) {
										return
									}
								}

								buf.Reset()
								buf.WriteString(curr[safeLen:])

								hasPartial = true

								break
							}
						}

						if !hasPartial {
							if !yield(curr, nil) {
								return
							}

							buf.Reset()
						}

						break
					} else {
						if openIdx > 0 {
							if !yield(curr[:openIdx], nil) {
								return
							}
						}

						insideThink = true
						curr = curr[openIdx+len("<think>"):]

						buf.Reset()
						buf.WriteString(curr)
					}
				} else {
					closeIdx := strings.Index(curr, "</think>")
					if closeIdx == -1 {
						buf.Reset()

						for i := 1; i < len("</think>"); i++ {
							if strings.HasSuffix(curr, "</think>"[:i]) {
								buf.WriteString(curr[len(curr)-i:])

								break
							}
						}

						break
					} else {
						insideThink = false
						curr = curr[closeIdx+len("</think>"):]

						buf.Reset()
						buf.WriteString(curr)
					}
				}
			}
		}

		if buf.Len() > 0 && !insideThink {
			_ = yield(buf.String(), nil)
		}
	}, nil
}
