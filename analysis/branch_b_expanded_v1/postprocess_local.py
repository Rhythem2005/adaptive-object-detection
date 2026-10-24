"""Serial postprocessing after replay completion; never overlaps scoring/timing."""
import os
import subprocess
import sys
import time
from analysis.branch_b_expanded_v1.continue_local import BASE,ROOT
from analysis.argoverse_hd_phase2.common import dump


def main():
    while not (ROOT/'EXECUTION_COMPLETE.json').exists():time.sleep(10)
    env=dict(os.environ,PYTHONPATH='extracted:.',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(ROOT/'analysis/matplotlib_cache'))
    commands=[
        [sys.executable,'-m','unittest','analysis.branch_b_expanded_v1.test_local','-v'],
        [sys.executable,'-u','-m','analysis.branch_b_expanded_v1.analyze_local','scoring'],
        [sys.executable,'-m','analysis.branch_b_expanded_v1.cases_local'],
        [sys.executable,'-m','analysis.branch_b_expanded_v1.figures_local']]
    node_bin = '/Users/rhythemsabharwalgmail.com/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
    tool_mjs = '/Users/rhythemsabharwalgmail.com/.cache/codex-runtimes/codex-primary-runtime/plugins/openai-primary-runtime/plugins/pdf/skills/pdf/container_tools/mark_artifact_operation_started.mjs'
    if os.path.exists(node_bin) and os.path.exists(tool_mjs):
        commands.append([node_bin, tool_mjs, '--operation-kind','edit','--expected-output-count','1','--output-format','pdf'])
    commands.extend([
        [sys.executable,'-m','analysis.branch_b_expanded_v1.report_local'],
        ['python3','-m','analysis.branch_b_expanded_v1.render_local'],
        [sys.executable,'-m','analysis.branch_b_expanded_v1.verify_local']])
    for i,cmd in enumerate(commands):
        print('POSTPROCESS',i,' '.join(cmd),flush=True)
        result=subprocess.run(cmd,cwd=BASE,env=env)
        dump(ROOT/f'analysis/postprocess_stage_{i}.json',dict(command=cmd,returncode=result.returncode))
        if result.returncode:raise RuntimeError(f'Postprocessing stage {i} failed; preserve outputs and inspect log')
    dump(ROOT/'POSTPROCESS_COMPLETE.json',dict(stages=len(commands),unix_time=time.time()))


if __name__=='__main__':main()
