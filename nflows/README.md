# ai as a reusable NFlow package

ai keeps provider protocols, HTTP requests, credentials, timeouts and normalized errors in Go, while NFlow defines orchestration and reusable workflow logic. Use NFlow composition rather than copying provider code into each application.

## Load the package

From a workflow in this checkout, use the local ai module:

```nflow
@require ../ as ai {
    provider: "openai",
    model: "gpt-4o-mini",
    api_key_env: "OPENAI_API_KEY",
    timeout: "30s"
}
```

After publishing the planned ai `v0.1.0` tag, downstream projects can use:

```nflow
@require github.com/nexssp/ai v0.1.0 as ai {
    provider: "openai",
    model: "gpt-4o-mini",
    api_key_env: "OPENAI_API_KEY",
    timeout: "30s"
}
```

The example uses a planned version; it will resolve only after that tag is published. ai pins the public Flow `v0.42.0`, which provides embedded bundle source modules, alias-qualified `@include`, and block/list post-run `@assert` syntax. The accepted `@require` options are `provider`, `model`, `api_key_env`, `timeout`, and `base_url`. `api_key_env` is the name of an environment variable, not the secret value. The default timeout is 60 seconds.

## Actions and reusable workflow

`ai.prompt` makes one completion and preserves existing Flow state fields. Its input supports `user`, optional `system`, `provider`, `model`, `base_url`, `max_tokens`, `temperature`, and `json_mode`; the result adds `content`, `model`, and token counts.

`ai.team` runs two or more declared role turns in order in a single Flow invocation. Each role can see the goal and earlier answers. It does not create persistent agents, sessions, or cross-run memory.

ai embeds and explicitly exports `nflows/lib/critic.nflow`. A consumer can import it from the same required bundle without copying the file:

```nflow
@include ai:nflows/lib/critic.nflow AS critic
{
    ...,
    goal: .goal,
    output: .output
}
-> critic.pipeline.review
```

The included pipeline calls `ai.prompt` from the same bundle. The public module pipeline is `critic.pipeline.review`; the local `nflows/critic.nflow` wrapper keeps the legacy `pipeline.si_ai_critic` entrypoint and also defines a `pipeline.review` alias; `roles.nflow` shows sequential roles, and `parallel_roles.nflow` shows independent concurrent prompts.

## Checks

The automated tests use fake providers and do not need a live API key:

```sh
go test ./...
task check
```

Build public Flow `v0.42.0` for the CLI integration test when needed. Install the repository hooks with `pre-commit install --hook-type pre-commit --hook-type pre-push`.
