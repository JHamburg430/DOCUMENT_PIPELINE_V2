-- Durable Agent-page failures, verified fixes, and later regression replays.
create table if not exists agent_repair_cases (
    request_id text primary key,
    source_run_id text not null unique,
    status text not null default 'saved_for_gateway_diagnosis'
        check (status in ('saved_for_gateway_diagnosis', 'diagnosing', 'fixed', 'verified', 'closed')),
    run_snapshot jsonb not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists agent_repair_fixes (
    id uuid primary key default gen_random_uuid(),
    request_id text not null references agent_repair_cases(request_id),
    root_cause text not null,
    fix_summary text not null,
    commit_sha text not null,
    regression_spec jsonb not null check (jsonb_typeof(regression_spec) = 'object'),
    verification_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);
create index if not exists agent_repair_fixes_case_created_idx
    on agent_repair_fixes (request_id, created_at desc);

create table if not exists agent_repair_regression_runs (
    id uuid primary key default gen_random_uuid(),
    fix_id uuid not null references agent_repair_fixes(id),
    source_revision text not null,
    status text not null check (status in ('passed', 'failed', 'blocked')),
    result_json jsonb not null default '{}'::jsonb,
    error text,
    created_at timestamptz not null default now()
);
create index if not exists agent_repair_regression_runs_fix_created_idx
    on agent_repair_regression_runs (fix_id, created_at desc);
