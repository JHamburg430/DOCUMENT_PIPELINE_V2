"""Opt-in owner diagnostic: exact worker, loaded model, bounded synthetic work.

Receipt attribution comes from the gateway's dispatch decision, never model
output. It is authenticated by the owner HTTPS response, not a portable signed
attestation. No inference/operator credential is accepted as owner authority.
"""
import threading
import time
import uuid
from urllib.request import build_opener

from .models import InferenceRequest
from .node_updater import NoRedirect
from .streaming import NDJSONResponse


APPLICATION = 'provis_employee_agent'
TIMEOUT = 10.0


class OwnerDiagnostic:
    def __init__(self, gateway, *, stream_factory=None, clock=time.monotonic):
        self.gateway, self.stream_factory, self.clock = gateway, stream_factory, clock
        self.slot = threading.BoundedSemaphore(1)  # no queue or parallel probes

    def _target(self, workspace, target, model):
        from .agent_link import AgentLink
        g = self.gateway
        record = next((r for r in g.enrollment.nodes() if r.node_id == target), None)
        if (record is None or record.enrollment_method != 'entra' or
                AgentLink._workspace({'tenant_id':record.identity_tenant,'subject':record.identity_subject}) != workspace):
            raise PermissionError('Target is outside owner workspace')
        approved = record.allowed_applications or g.node_default_applications
        if APPLICATION not in approved or target not in g.authorized_worker_ids(APPLICATION):
            raise PermissionError('Target is not approved for diagnostic application')
        node = next((n for n in g.scheduler.snapshots() if n.node_id == target), None)
        if (node is None or not node.healthy or not node.inference_enabled or
                not 0 <= time.time()-node.observed_at_epoch <= g.scheduler.policy.heartbeat_ttl_seconds or
                APPLICATION not in node.allowed_applications or node.active_requests or node.queued_requests):
            raise PermissionError('Target is unavailable, stale or busy')
        if 'diagnostic_loaded_only_v1' not in node.capabilities:
            raise PermissionError('Target needs loaded-only diagnostic support')
        if model not in node.installed_models or not any(m.name == model for m in node.loaded_models):
            raise PermissionError('Diagnostic requires an already installed and loaded model')
        if not node.node_app_version:
            raise PermissionError('Target version unavailable')
        if g.node_secrets.get(target) != record.secret.encode() or not g.node_agent_urls.get(target):
            raise PermissionError('Current signed worker transport required')
        return node

    def run(self, workspace, source_node, payload, *, cancelled=None):
        from .gateway import node_request_signature
        if type(payload) is not dict or set(payload) != {'target_node_id','model'}:
            raise ValueError('Only exact target and model are accepted')
        target,model = payload['target_node_id'],payload['model']
        if any(type(v) is not str or not 1 <= len(v) <= 128 for v in (target,model)):
            raise ValueError('Bounded target and model required')
        spec=self.gateway.model_catalog.get(model)
        if spec is None or spec.backend != 'ollama' or 'chat' not in spec.capabilities:
            raise PermissionError('Catalogued Ollama chat model required')
        if not self.slot.acquire(blocking=False): raise RuntimeError('Diagnostic already in progress')
        cancelled=cancelled if cancelled is not None else threading.Event()
        deadline=self.clock()+TIMEOUT
        decision=stream=watcher=None
        stopped=threading.Event()
        request=InferenceRequest('owner-diagnostic:'+str(uuid.uuid4()),APPLICATION,'owner_diagnostic',model,
                                 spec.size_bytes,spec.min_vram_mb,context_tokens=8192,max_latency_ms=10000,
                                 required_capabilities=('chat',),allowed_node_ids=(target,),backend=spec.backend,
                                 min_gpu_count=spec.min_gpu_count,allow_multi_gpu=spec.allow_multi_gpu)
        try:
            self._target(workspace,target,model)
            # Deliberately do not call _authoritative_request, which replaces
            # caller node restrictions. This server-owned path narrows the
            # already-checked application allowlist to exactly one worker.
            decision=self.gateway.scheduler.choose_and_reserve(request)
            if not decision.admitted or decision.node_id != target:
                raise PermissionError('Exact diagnostic target unavailable')
            if decision.requires_model_pull or decision.requires_model_warm or decision.eviction_candidates:
                raise PermissionError('Diagnostic model preparation forbidden')
            node=self._target(workspace,target,model)  # recheck after reservation
            if cancelled.is_set(): raise ConnectionAbortedError('Diagnostic caller disconnected')
            envelope={'operation':'generate','request_id':request.request_id,'issued_at':time.time(),
                      'diagnostic_loaded_only':True,
                      'payload':{'model':model,'prompt':'Reply PROVIS_DIAGNOSTIC_OK only.', 'think':False,
                                 'stream':True,'options':{'num_predict':8,'temperature':0}}}
            if node.gpu_placement_mode in ('single_gpu','multi_gpu'):
                envelope['placement']=decision.placement.to_dict() if decision.placement else None
            envelope['signature']=node_request_signature(envelope,self.gateway.node_secrets[target])
            remaining=deadline-self.clock()
            if remaining <= 0: raise TimeoutError('Diagnostic deadline exceeded')
            url=self.gateway.node_agent_urls[target].rstrip('/')+'/v1/node/inference'
            if self.stream_factory:
                stream=self.stream_factory(url,envelope,timeout=remaining)
            else:
                stream=NDJSONResponse(url,envelope,timeout=remaining,opener=build_opener(NoRedirect()).open)
            def watch():
                while not stopped.wait(.05):
                    if cancelled.is_set():
                        stream.cancel();return
            watcher=threading.Thread(target=watch,daemon=True,name='owner-diagnostic-cancel');watcher.start()
            output='';done=False
            for index,event in enumerate(stream):
                if cancelled.is_set(): raise ConnectionAbortedError('Diagnostic caller disconnected')
                if self.clock()>=deadline: raise TimeoutError('Diagnostic deadline exceeded')
                if index>=64: raise ValueError('Diagnostic event limit exceeded')
                data=event.get('response',event)
                if type(data) is not dict or event.get('error') or data.get('error'):
                    raise ValueError('Diagnostic worker failed')
                token=data.get('response','')
                if type(token) is not str or len(output)+len(token)>512: raise ValueError('Diagnostic output limit exceeded')
                output+=token
                if data.get('done') is True:
                    done=True;break
            if not done: raise ValueError('Diagnostic completion missing')
            if cancelled.is_set(): raise ConnectionAbortedError('Diagnostic caller disconnected')
            return {'receipt':{'request_id':request.request_id,'source_node_id':source_node,
                               'target_node_id':target,'node_id':decision.node_id,'model':model,
                               'node_version':node.node_app_version,'completed':True},'response':output}
        except (PermissionError,ValueError,TimeoutError,ConnectionAbortedError):
            raise
        except Exception:
            raise RuntimeError('Diagnostic worker transport failed') from None
        finally:
            stopped.set()
            try:
                if stream is not None: stream.close()
            finally:
                if watcher is not None: watcher.join(timeout=.2)
                try:
                    if decision is not None and decision.admitted:
                        self.gateway.scheduler.release(decision.node_id,request.request_id)
                finally: self.slot.release()
