alter table agent_repair_cases
    add column if not exists gateway_session_key text,
    add column if not exists gateway_handoff_status text,
    add column if not exists gateway_handoff_error text;

do $$ begin
    if not exists (select 1 from pg_constraint where conname = 'agent_repair_cases_gateway_handoff_status_check') then
        alter table agent_repair_cases add constraint agent_repair_cases_gateway_handoff_status_check
            check (gateway_handoff_status in ('pending', 'sent', 'failed'));
    end if;
end $$;
