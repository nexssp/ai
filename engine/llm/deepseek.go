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
		cfg.Name = providerDeepSeek
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

type deepSeekStreamFilter struct {
	insideThink bool
	buffer      strings.Builder
}

func (f *deepSeekStreamFilter) consume(chunk string, yield func(string, error) bool) bool {
	f.buffer.WriteString(chunk)
	curr := f.buffer.String()
	for curr != "" {
		var done bool
		if f.insideThink {
			curr, done = f.consumeThinking(curr)
		} else {
			var ok bool
			curr, done, ok = f.consumeVisible(curr, yield)
			if !ok {
				return false
			}
		}
		if done {
			return true
		}
	}
	return true
}

func (f *deepSeekStreamFilter) consumeVisible(curr string, yield func(string, error) bool) (remaining string, done, ok bool) {
	openIdx := strings.Index(curr, thinkOpenTag)
	if openIdx == -1 {
		for i := 1; i < len(thinkOpenTag); i++ {
			if !strings.HasSuffix(curr, thinkOpenTag[:i]) {
				continue
			}
			safeLen := len(curr) - i
			if safeLen > 0 && !yield(curr[:safeLen], nil) {
				return "", true, false
			}
			f.buffer.Reset()
			f.buffer.WriteString(curr[safeLen:])
			return "", true, true
		}
		if !yield(curr, nil) {
			return "", true, false
		}
		f.buffer.Reset()
		return "", true, true
	}
	if openIdx > 0 && !yield(curr[:openIdx], nil) {
		return "", true, false
	}

	f.insideThink = true
	curr = curr[openIdx+len(thinkOpenTag):]
	f.buffer.Reset()
	f.buffer.WriteString(curr)
	return curr, false, true
}

func (f *deepSeekStreamFilter) consumeThinking(curr string) (remaining string, done bool) {
	closeIdx := strings.Index(curr, thinkCloseTag)
	if closeIdx == -1 {
		f.buffer.Reset()
		for i := 1; i < len(thinkCloseTag); i++ {
			if strings.HasSuffix(curr, thinkCloseTag[:i]) {
				f.buffer.WriteString(curr[len(curr)-i:])
				break
			}
		}
		return "", true
	}

	f.insideThink = false
	curr = curr[closeIdx+len(thinkCloseTag):]
	f.buffer.Reset()
	f.buffer.WriteString(curr)
	return curr, false
}

func (f *deepSeekStreamFilter) flush(yield func(string, error) bool) {
	if f.buffer.Len() == 0 || f.insideThink {
		return
	}
	_ = yield(f.buffer.String(), nil)
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
		var filter deepSeekStreamFilter
		for chunk, streamErr := range rawStream {
			if streamErr != nil {
				yield("", streamErr)
				return
			}
			if !filter.consume(chunk, yield) {
				return
			}
		}
		filter.flush(yield)
	}, nil
}
