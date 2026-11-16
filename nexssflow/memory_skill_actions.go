package nexssflow

import (
	"context"

	"github.com/nexssp/kernel/action"
	"github.com/nexssp/kernel/xerr"

	"github.com/nexssp/ai/memory"
	"github.com/nexssp/ai/skills"
)

// SkillGetRequest selects one host-approved skill by name.
type SkillGetRequest struct {
	Name string `json:"name"`
}

type flowMemoryScopeRequest struct {
	UserID         string `json:"user_id"`
	ConversationID string `json:"conversation_id"`
	MemoryScope    string `json:"memory_scope,omitempty"`
}

type flowMemoryReplyRequest struct {
	Scope        flowMemoryScopeRequest `json:"scope"`
	UserMessage  string                 `json:"user_message"`
	SystemPrompt string                 `json:"system_prompt,omitempty"`
	Remember     bool                   `json:"remember,omitempty"`
}

func (request flowMemoryReplyRequest) toDomain() memory.Request {
	return memory.Request{
		Scope: memory.Scope{
			UserID:         request.Scope.UserID,
			ConversationID: request.Scope.ConversationID,
			MemoryScope:    memory.Level(request.Scope.MemoryScope),
		},
		UserMessage:  request.UserMessage,
		SystemPrompt: request.SystemPrompt,
		Remember:     request.Remember,
	}
}

func newMemorySkillActions(memoryService *memory.Service, catalog *skills.Catalog) []action.AnyAction {
	return []action.AnyAction{
		action.New[flowMemoryReplyRequest, memory.Response]("ai.memory.reply", func(ctx context.Context, request flowMemoryReplyRequest) (memory.Response, error) {
			if memoryService == nil {
				return memory.Response{}, xerr.Internal("ai.memory.reply: memory service is not configured")
			}
			return memoryService.Reply(ctx, request.toDomain())
		}).Description("Reply using conversation-local working history and an explicitly selected conversation or user long-term memory scope").Tag("ai", "memory", "reply").Build(),
		action.New[struct{}, []skills.Skill]("ai.skills.list", func(context.Context, struct{}) ([]skills.Skill, error) {
			if catalog == nil {
				return nil, xerr.Internal("ai.skills.list: approved skills catalog is not configured")
			}
			return catalog.List(), nil
		}).Description("List skills loaded from the host-approved directory; skill text is never executed").Tag("ai", "skills", "list").Build(),
		action.New[SkillGetRequest, skills.Skill]("ai.skills.get", func(_ context.Context, request SkillGetRequest) (skills.Skill, error) {
			if catalog == nil {
				return skills.Skill{}, xerr.Internal("ai.skills.get: approved skills catalog is not configured")
			}
			return catalog.Get(request.Name)
		}).Description("Read one host-approved skill from the startup catalog; arbitrary paths and unapproved drafts are not accessible").Tag("ai", "skills", "get").Build(),
		action.New[skills.Skill, skills.DraftReceipt]("ai.skills.draft", func(_ context.Context, draft skills.Skill) (skills.DraftReceipt, error) {
			if catalog == nil {
				return skills.DraftReceipt{}, xerr.Internal("ai.skills.draft: skills catalog is not configured")
			}
			return catalog.CreateDraft(draft)
		}).Description("Validate and write a skill proposal as inert JSON outside the active catalog; host review and activation are separate").Tag("ai", "skills", "draft").Build(),
	}
}
