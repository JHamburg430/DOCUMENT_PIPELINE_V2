"""Server-owned worker membership derived from current company enrollment."""


class EnrolledWorkerAdmission:
    def __init__(self, tenant_id):
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError('Company enrollment tenant required')
        self.tenant_id = tenant_id.strip().casefold()

    def allowed_node_ids(self, gateway, application):
        eligible = []
        for node in gateway.enrollment.nodes():
            if (node.enrollment_method != 'entra' or
                    not isinstance(node.identity_subject, str) or not node.identity_subject.strip() or
                    not isinstance(node.identity_tenant, str) or
                    node.identity_tenant.casefold() != self.tenant_id):
                continue
            approved = node.allowed_applications or gateway.node_default_applications
            if application not in approved:
                continue
            # Stale cached credentials/endpoints cannot authorize a replacement
            # or revoked worker. Heartbeat and scheduler checks still apply.
            if (gateway.node_secrets.get(node.node_id) != node.secret.encode() or
                    gateway.node_agent_urls.get(node.node_id) != node.agent_url):
                continue
            eligible.append(node.node_id)
        return tuple(sorted(eligible))


def configure_worker_admission(gateway, mode):
    if mode == 'static':
        gateway.enrolled_worker_admission = None
        return
    if mode != 'entra_enrolled':
        raise ValueError('Unsupported worker admission mode')
    if (not gateway.require_entra_enrollment or gateway.enrollment is None or
            gateway.entra_validator is None or not gateway.require_node_agents or
            not gateway.require_application_identity or
            not gateway.scheduler.policy.require_application_allowlist):
        raise ValueError('Automatic workers require authenticated company enrollment and application policy')
    gateway.enrolled_worker_admission = EnrolledWorkerAdmission(gateway.entra_validator.config.tenant_id)
