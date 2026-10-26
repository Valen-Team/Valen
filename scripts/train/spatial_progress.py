"""Render a live Markdown checklist from validated data and actual pipeline artifacts."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from scripts.train.monitor_spatial import last_metric
from scripts.train.run_spatial_pipeline import required_benchmarks
from scripts.train.summarize_spatial import collect_results
from valen.training.checkpoint import file_sha256


def read_json(path):
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def checked(done, text, status="待办", indent=""):
    return f"{indent}- [{'x' if done else ' '}] {'已完成' if done else status}：{text}"


def data_progress(control, task, data_root):
    if task == "seg" and (Path(data_root) / "seg/manifests/train_refcoco.json").is_file():
        manifest = read_json(Path(data_root) / "seg/manifests/train.json")
        return (f"{manifest['samples']:,} 条已转换，等待完整校验" if manifest.get("samples")
                else "RefCOCO 80,000 条已完成；PhraseCut 20,000 条转换与校验中")
    for filename in ("prepare-natural.log", "prepare-phrasecut.log"):
        path = control / filename
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 32768))
            rows = stream.read().splitlines()
        for row in reversed(rows):
            try:
                value = json.loads(row)
            except (ValueError, UnicodeDecodeError):
                continue
            if Path(value.get("root", ".")) == Path(data_root) / task and value.get("total"):
                return f"图片 {value['images_ready']:,}/{value['total']:,}（{value['images_ready']/value['total']:.1%}）"
    return "等待训练媒体、转换与校验完成"


def stage_progress(run, stage, available, state, deferred=False, expected_records=100000):
    completion = read_json(run / stage / "training_complete.json")
    metric = last_metric(run / stage / "metrics.jsonl")
    done = (completion.get("stop_reason") == "epochs_complete" and completion.get("epoch") == 1
            and bool(metric) and metric.get("seen") == metric.get("total_records") == expected_records)
    if metric:
        detail = f"{metric['seen']:,}/{metric['total_records']:,} 条（{metric['seen']/metric['total_records']:.1%}），step {metric['step']:,}"
        return done, detail, "暂缓（保留 checkpoint）" if deferred else "进行中" if state == "running" else "待恢复/检查"
    return done, f"1 epoch、{expected_records:,} 条、固定 8 卡", "等待前序任务" if available else "待办"


def write_progress(state, registry, data_root, model_root, output):
    registry, data_root, model_root, output = map(Path, (registry, data_root, model_root, output))
    control = registry.parent
    results = collect_results(state, data_root, model_root)
    by_name = {model["name"]: model for model in results["models"]}
    gui2_complete = by_name.get("Valen-Decision-GUI-2B", {}).get("deliverables_ready", False)
    gui_release = state.get("release_gui2_after_training", {})
    gui_training_released = bool(gui_release.get("released_utc"))
    pipelines = [job for job in state["jobs"] if job.get("kind") == "pipeline"]
    originals = [job for job in pipelines if job.get("variant") not in {"full", "rlcd"}]
    full = [job for job in pipelines if job.get("variant") == "full"]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+8")
    running, requested = state.get("running_gpus", 0), state.get("reserved_gpus", 0)
    lines = ["# Valen 专项训练进度清单", "", f"更新时间：{stamp}", "",
        "`[x]` 已完成；`[ ] 进行中` 正在执行；`[ ] 待办/等待前序任务` 尚未完成。",
        "本文件由实际数据校验、checkpoint、导出与评测完成记录自动更新；排队或生成目录不计为完成。", "",
        f"资源：已获配 {running} 卡，运行加排队共申请 {requested} 卡，上限 {state.get('gpu_limit', 64)} 卡。集群状态检查：{state.get('last_check_utc', '待首次检查')}。", "",
        f"计划：原定 {len(originals)} 个模型，新增 {len(full)} 组全量微调对照；当前有效范围共 {results['expected_models']} 个版本，已验收 {results['completed_models']} 个。", "", "## 数据与基础流程", ""]
    if state.get("det_referring_revision"):
        lines[10:10] = ["本轮 Det 仅使用 RefCOCO / RefCOCO+ / RefCOCOg 与 LISA/ReasonSeg，合计 100k；COCO 类别检测样本与评测已移除。",
                        ("当前全部训练暂停，仅完成GUI数据验收，保留已有模型和恢复点。" if state.get("user_pause") else
                         ("GUI 2B训练与评测已完成，Det与Seg继续暂停，保留已有模型和恢复点。" if gui2_complete else
                          "当前仅推进GUI 2B训练与评测，Det与Seg暂停，保留已有模型和恢复点。") if state.get("gui2_lora64_revision") else
                         "2B Full64对照已完成；当前复用64卡推进LoRA-RLCD，其他训练保留恢复点。" if state.get("det_rlcd_plan", {}).get("serial_order") else
                         "原16卡训练已暂停，2B从共享预热头以64卡重新训练；其他尺寸保留恢复点。" if state.get("det2_full64_revision") else
                         "首批 0.8B、2B、4B 各 16 卡并行，9B 与 Full 按依赖接续；Seg 保留 checkpoint，等待 Det 优先队列后恢复。"), ""]
    if state.get("det_rlcd_plan"):
        parent = "合并LoRA" if state["det_rlcd_plan"].get("initialization_variant") == "lora_merged" else "Full"
        tuning = "LoRA" if state["det_rlcd_plan"].get("rl_finetuning_type") == "lora" else "全参数"
        sizes="/".join(state["det_rlcd_plan"]["sizes"])
        lines[10:10] = [f"Det {sizes} RLCD：从各自{parent}回放原100k重新筛错例，再做{tuning}强化；0.8B/4B RLCD已按用户要求取消。" if state["det_rlcd_plan"].get("canceled_sizes") else
                        f"新增 Det 0.8B/2B/4B RLCD 三组；从各自{parent}回放原100k重新筛错例，再做{tuning}强化，合计计划22个模型。", ""]
    if not state.get("user_pause") and set(state.get("suspended_tasks", [])) >= {"seg", "gui"}:
        lines[10:10] = ["当前全部卡组优先 Det；Seg、GUI 训练与评测统一暂缓，保留后续队列及已有产物。", ""]
    if state.get("gui2_lora64_revision"):
        revision=state["gui2_lora64_revision"]
        sample_label = f"{revision.get('training_samples', 100000):,}"
        data_accepted = read_json(control / "gui-data-172k-training-accepted.json").get("validated") is True
        lines[10:10]=[(f"GUI {sample_label}条数据已验收，训练保持暂停；已获配64卡继续保留。" if state.get("user_pause") and data_accepted else
                      f"GUI {sample_label}条仅继续数据构建与验收，训练保持暂停；已获配64卡继续保留。" if state.get("user_pause") else
                      f"GUI 2B新版{sample_label}条已完成训练、导出、全部评测及示例验收；按用户要求释放GPU。" if gui2_complete and state.get('release_gui2_after_training') else
                      f"GUI 2B新版{sample_label}条已完成训练、导出、全部评测及示例验收；已获配64卡继续保留。" if gui2_complete else
                      f"GUI 2B新版{sample_label}条已完成64卡训练与导出，原训练卡已释放；等待完成独立8卡评测。" if gui_training_released else
                      f"当前仅推进GUI 2B新版{sample_label}条：64卡、全局batch128；任务头预热后进行LoRA联合训练，各1epoch。"),
                      f"训练数据目录：{revision['data_root']}；安装与校验状态以数据清单为准。", ""]
    elif state.get("det2_full64_revision"):
        lines[10:10] = [("Det LoRA-RLCD复用同一组64卡串行推进：2B完整评测 → 0.8B筛错/训练/评测 → 4B筛错/训练/评测；全局batch128。"
                        if state.get("det_rlcd_plan", {}).get("serial_order") else
                        "当前推进 Det 2B 合并LoRA来源的RLCD：64卡、全局batch128；Full对照已完成，其他Det暂停并保留恢复点。"
                        if state.get("det2_lora_rlcd_revision") else
                        "当前仅推进 Det 2B Full：64卡、全局batch128，从共享预热头重新训练100k（约782步）；其余Det暂停并保留恢复点。"), ""]
    if state.get("coco_prompt_revision") or state.get("det_referring_revision"):
        if state.get("coco_prompt_revision"):
            lines[10:10] = [("本轮 Det 已切换 COCO 单类查询：训练每图一次，评测逐类轮流查询；四尺寸及两个 Full 对照重新训练。"
                         if state["coco_prompt_revision"].get("protocol") == "coco_single_category_v3" else
                         "本轮 Det 已切换 COCO prompt v2，四尺寸及两个 Full 对照全部重新训练；旧版完成状态不计入本轮。"),
                        "旧版 Det 权重/缓存清理与修正版训练的状态以本清单和实际产物为准。", ""]
        cleanup = read_json(control / "det-legacy-cleanup.json")
        lines.append(checked(cleanup.get("verified") is True, "清理过期 Det 权重、优化器及旧训练清单，保留精简历史指标", "迁移中"))
    if state.get("retired_training_checkpoint_cleanup"):
        cleanup = read_json(Path(state["retired_training_checkpoint_cleanup"]))
        cleanup_done = (cleanup.get("status") == "completed" and cleanup.get("applied") is True
                        and cleanup.get("all_protected_files_unchanged") is True
                        and bool(cleanup.get("files"))
                        and all(not Path(row["path"]).exists() for row in cleanup["files"]))
        lines.append(checked(cleanup_done,
            f"清理已取消/替代训练检查点（{cleanup.get('removed_unique_file_bytes', 0)/1024**3:.2f} GiB），保留暂停版本恢复点"))
    smoke = read_json(control / "gpu-smoke-2n/complete.json")
    lines.append(checked(smoke.get("ranks") == 8 and set(smoke.get("tasks", {})) == {"det", "seg", "gui"},
                         "Det、Seg、GUI 真实八卡小样本训练与导出流程验证"))
    lines.append(checked((control / "gpu-smoke-2n/independent-process-checks.json").is_file(),
                         "导出模型在独立进程加载与推理验证"))
    if state.get("execution_groups"):
        pilot = read_json(control / "referring-v4-16gpu-smoke/pipeline_complete.json")
        verified = (pilot.get("world_size") == 16 and len(set(pilot.get("gpu_uuids", []))) == 16
                    and pilot.get("resumed_parameter_updates_verified") is True and pilot.get("global_batch") == 32)
        lines.append(checked(verified, "Det 真实 16 卡训练及断点恢复：四节点、完整优化器/RNG、全局 batch 32", "验证中"))
    data_protocol = read_json(data_root / "det/manifests/ready.json").get("protocol")
    conditional_data = data_protocol in {"coco_known_categories_v2", "coco_single_category_v3"}
    referring_data = data_protocol == "det_referring_reasonseg_v4"
    def evaluation_manifest(task, name):
        base = data_root / "eval" / task
        if name == "coco2017_conditional_val2017":
            base = base / "protocols/coco_prompt"
        return base / "manifests" / (name + ".json")
    suites = {task: required_benchmarks(task, "none" if referring_data else "conditional" if conditional_data else "standard")
              for task in ("det", "seg", "gui")}
    eval_done = all(evaluation_manifest(task, name).is_file()
                    for task in ("det", "seg", "gui")
                    for name in suites[task])
    lines.append(checked(eval_done, f"{sum(map(len, suites.values()))} 个必测 benchmark/split 的评估数据准备（ReasonSeg 另列）"))
    for task in ("det", "seg", "gui"):
        seal = read_json(data_root / task / "manifests/ready.json")
        if task == "det" and state.get("coco_prompt_revision") and seal.get("protocol") not in {"coco_known_categories_v2", "coco_single_category_v3"}:
            seal = read_json(data_root / "det/manifests/ready_coco_prompt_v2.json")
        expected_samples = state.get('gui2_lora64_revision', {}).get('training_samples', 100000) if task == 'gui' else 100000
        done = seal.get("validated") is True and seal.get("samples") == expected_samples
        detail = (f"{seal['samples']:,} 条、{seal['images']:,} 张图，独立媒体及全局去重校验通过" if done
                  else data_progress(control, task, data_root))
        lines.append(checked(done, f"{task.upper()} 训练数据：{detail}", "进行中" if task == "det" else "等待前序任务"))
    gui_revision = read_json(control / state.get("gui_data_revision_file", "gui-data-revision-v2.json"))
    if gui_revision:
        revision_root = Path(gui_revision.get("staging_root", "."))
        revised_seal = read_json(revision_root / "gui/manifests/ready.json")
        target_samples = gui_revision.get("target_samples", 100000)
        install = read_json(control / gui_revision.get("install_report_file", "gui-data-v2-install.json"))
        formal_seal = read_json(data_root / "gui/manifests/ready.json")
        installed = (install.get("applied") is True and install.get("status") == "installed_and_verified"
                     and formal_seal.get("validated") is True and formal_seal.get("samples") == target_samples
                     and bool(install.get("records_sha256"))
                     and install["records_sha256"] == formal_seal.get("records_sha256"))
        prepared = installed or (revised_seal.get("validated") is True
                                 and revised_seal.get("samples") == target_samples)
        lines.extend(["", "## 增补：GUI 专业桌面数据", "",
            checked(prepared, f"构建并校验新版 GUI {target_samples:,}条：专业桌面、小控件与原始真实框", "进行中"),
            checked(installed, "替换正式 gui 与 eval/gui 开发数据，清理旧 GUI 冗余副本",
                    "等待最终数据验收与正式替换" if gui_revision.get("write_access_verified") else "等待正式目录写入权限"),
            checked(gui2_complete, f"GUI 2B使用新版{target_samples:,}条完成LoRA训练、导出、三个完整benchmark及推理示例验收", "等待模型验收"),
            "Det 数据和原有模型保持不变；新版 GUI 的完成状态以对应运行产物为准。", ""])
    if state.get("gui_additional_benchmarks"):
        extra = state['gui_additional_benchmarks']
        extra_root = Path(extra.get('data_root', data_root / 'eval/gui/additional'))
        extra_seal = read_json(extra_root / 'manifests/ready.json')
        extra_complete = by_name.get('Valen-Decision-GUI-2B', {}).get('checks', {}).get('additional_gui_benchmarks', False)
        lines.extend(['', '## GUI 新增 grounding benchmark', '',
            checked(extra_seal.get('validated') is True,
                    '准备官方完整UI-Vision 5,479条、OSWorld-G refined 564条、MMBench-GUI L2 3,594条，并完成训练重叠审计', '进行中'),
            checked(extra_complete,'172k GUI 2B训练后补测全部9,637条，报告官方分项/汇总及重叠情况', '训练完成后接续'),
            '保留三个ScreenSpot完整测试；新增测试不参与选模，不将在线交互成绩混作grounding成绩。', ''])
    if state.get('release_gui2_after_training'):
        policy=state['release_gui2_after_training']
        status=policy.get('status','armed')
        trained=read_json(Path(policy['run_dir'])/'training_export_complete.json')
        lines.extend(['', '## GUI 训练后评测与资源释放', '',
            checked(bool(trained),'64卡完成2B训练、导出及独立重载校验；评测与训练租约分离','训练中/待启动'),
            checked(bool(policy.get('released_utc')),'立即释放原64卡训练租约，并撤销闲置排队申请','训练导出完成后执行'),
            checked(status=='evaluated_and_all_leases_released','另开8卡完成三个ScreenSpot及新增9,637条评测，完成后自动释放',
                    '评测已提交（独立8卡）' if policy.get('evaluation_job_id') else '等待训练与数据验收'),
            f'当前回收/评测状态：{status}。不接续其他模型训练。',''])
    if conditional_data:
        dev_root = data_root / "eval/det/protocols/coco_prompt"
        dev = read_json(dev_root / "manifests/dev.json")
        seal = read_json(data_root / "det/manifests/ready.json")
        dev_done = (dev.get("validated") is True and dev.get("independent_media") is True
                    and dev.get("records_sha256") == seal.get("development_sha256")
                    and (dev_root / "processed/dev.jsonl").is_file())
        lines.append(checked(dev_done, "Det COCO/RefCOCO 开发集独立归档至 eval/det，保持封存校验值", "迁移中"))
    if referring_data:
        dev = read_json(data_root / "eval/det/manifests/dev.json")
        seal = read_json(data_root / "det/manifests/ready.json")
        lines.append(checked(dev.get("samples") == 712 and dev.get("records_sha256") == seal.get("development_sha256"),
            "Det 开发集：512 条 RefCOCO 系列 + 完整 200 条 ReasonSeg val，独立保存在 eval/det"))
    resource_text = (f"配置卡组复用；训练全部暂停，保留现有64卡，总上限 {state.get('gpu_limit', 64)} 卡" if state.get("user_pause") else
                     f"训练与评测卡组均已释放；其他训练暂停，总上限 {state.get('gpu_limit', 64)} 卡"
                     if gui_release.get('status') == 'evaluated_and_all_leases_released' else
                     f"原64卡训练租约已释放；独立评测申请8卡，其他训练暂停，总上限 {state.get('gpu_limit', 64)} 卡"
                     if gui_training_released else
                     f"配置卡组复用；当前GUI 2B LoRA独占64卡，其他训练暂停，总上限 {state.get('gpu_limit', 64)} 卡"
                     if state.get("gui2_lora64_revision") else
                     f"配置卡组复用；LoRA-RLCD每模型64卡，按2B → 0.8B → 4B串行，总上限 {state.get('gpu_limit', 64)} 卡"
                     if state.get("det_rlcd_plan", {}).get("serial_order") else
                     f"配置卡组复用；当前2B Full独占64卡，其他训练暂停，总上限 {state.get('gpu_limit', 64)} 卡"
                     if state.get("det2_full64_revision") else
                     f"配置卡组复用；Det 每模型 16 卡、GUI/Seg 每模型 8 卡、总上限 {state.get('gpu_limit', 64)} 卡"
                     if state.get("execution_groups") else
                     f"配置卡组复用；每模型 8 卡、总上限 {state.get('gpu_limit', 64)} 卡")
    lines.append(checked(state.get("scheduling_policy") in {"reuse_eight_gpu_allocations", "reuse_reserved_gpu_allocations"}, resource_text))
    monitor = read_json(control / state.get("monitor_report", "monitor-latest.json"))
    lines.append(checked(monitor.get("interval_seconds") == 900, "启用每 15 分钟巡检"))
    lines.append(checked(state.get("full_comparisons", {}).get("export_precision_verified", False),
                         "全量微调对照流程与 FP32 权重/BF16 计算精度的导出验证", "进行中"))
    for barrier in state.get("evaluation_barriers", []):
        if "rlcd" in barrier.get("expected_variants", {}).values():
            description = "、".join(barrier["models"]) + " 完整11项评测及训练集回放验收后，放行下一尺寸RLCD"
        elif "full" in barrier.get("expected_variants", {}).values():
            description = "Det 2B Full64与LoRA完成全部11项评测及性能/成本对照"
        else:
            description = ("Det 三组 LoRA（0.8B、2B、4B）完成 RefCOCO 系列、ReasonSeg"
                           + (" 与 GroundingME" if state.get("groundingme_support") else "")
                           + " 完整基线评测")
        lines.append(checked(barrier.get("ready") is True,
            description,
            "等待完整评测"))
    if state.get("gui_diagnosis"):
        audit = read_json(control / "gui-pro-diagnosis/data-audit.json")
        manifest = read_json(data_root / "eval/gui/manifests/screenspot_pro_test.json")
        verified = (bool(manifest) and audit.get("records_sha256") == manifest.get("records_sha256")
                    and audit.get("records") == manifest.get("samples")
                    and audit.get("media_match_official_bytes") is True
                    and audit.get("official_gold_center_accuracy") == 1
                    and audit.get("local_official_point_scorer_agreement") == 1)
        diagnostic = Path(state["gui_diagnosis"]["run_dir"])
        marker, summary = read_json(diagnostic / "pipeline_complete.json"), read_json(diagnostic / "paired_summary.json")
        expected = state["gui_diagnosis"]["expected_models"]
        expected_rows = {(("Qwen3.5-" if mode == "native" else "Valen-Decision-GUI-") + size, mode, benchmark, pixels)
                         for size in ("0.8B", "2B", "4B", "9B") for mode in ("native", "spatial")
                         for benchmark in ("screenspot_pro_test", "screenspot_v2_test") for pixels in (1048576, 4194304)}
        table = summary.get("table", [])
        observed_rows = {(row.get("model"), row.get("mode"), row.get("benchmark"), row.get("max_pixels")) for row in table}
        paired = (marker.get("models") == summary.get("models") == expected
                  and marker.get("cases_per_model") == summary.get("cases_per_model")
                  == state["gui_diagnosis"]["cases_per_model"] and observed_rows == expected_rows
                  and len(table) == len(expected_rows) and all(row.get("samples") == 32 for row in table))
        activity = "等待现有卡组接续"
        for allocation in state.get("allocations", []):
            for path in Path(allocation["queue"]).parent.glob("worker.node-*.json"):
                worker = read_json(path)
                if state["gui_diagnosis"].get("task_name") and worker.get("task") == state["gui_diagnosis"]["task_name"]:
                    activity = "需处理" if worker.get("state") == "blocked" else "进行中"
        lines.extend(["", "## 增补：GUI ScreenSpot-Pro 低分排查", "",
            checked(verified, "全部官方标注、截图校验与点击计分回放"),
            checked(paired, "原生基座/空间头四尺寸、固定样本与 1MP/4MP 的同八卡对照；诊断结果独立于正式成绩", activity)])
    reasonseg_jobs = [job for job in pipelines if job["task"] in {"det", "seg"}]
    reasonseg_enabled = any((data_root / task / "manifests/reasonseg.json").is_file() for task in ("det", "seg"))
    if reasonseg_enabled:
        lines.extend(["", "## 增补：LISA / ReasonSeg" if referring_data else "## 增补：COCO 复核与 LISA / ReasonSeg", ""])
        gold_dir = ("coco-single-category-v3-gold" if data_protocol == "coco_single_category_v3" else
                    "coco-prompt-v2-gold" if conditional_data else "coco-audit")
        gold = read_json(control / gold_dir / "gold-audit.json")
        gold_done = gold.get("perfect_prediction_metrics", {}).get("AP") == 1
        if conditional_data:
            manifest = read_json(data_root / "eval/det/protocols/coco_prompt/manifests/coco2017_conditional_val2017.json")
            gold_done = (gold_done and bool(manifest) and gold.get("records_sha256") == manifest.get("records_sha256")
                         and gold.get("protocol") == "known_category_conditional_detection"
                         and gold.get("images") == manifest.get("catalog_images")
                         and gold.get("queried_images") == manifest.get("images")
                         and gold.get("prompts") == manifest.get("samples"))
        if not referring_data:
            lines.append(checked(gold_done,
                                 "COCO 全集真实框回放：坐标、类别映射和官方 AP 满分校验"))
        diagnosed = ((control / "coco-audit/prediction-audit-complete.json").is_file() or
                     (control / "coco-audit-cpu/Valen-Decision-Det-0.8B/diagnostics.json").is_file())
        if not referring_data:
            lines.append(checked(diagnosed, "旧版 COCO 实际预测抽样诊断：定位召回、分类与重复框（非最终成绩）", "排队/执行中"))
        if conditional_data:
            revision = read_json(data_root / "det/manifests/ready.json")
            lines.append(checked(revision.get("validated") is True,
                f"启用 COCO 条件检测新版 100k：每条 {revision.get('max_categories', 3)} 类，{revision.get('coco_train_images', 0):,} 张 COCO 训练图，"
                f"{revision.get('coco_images_with_second_prompt', 0):,} 张重复一次；独立 COCO 开发图 512 张"))
        for task in ("det", "seg"):
            manifest = read_json(data_root / task / "manifests/reasonseg.json")
            splits = manifest.get("splits", {})
            ready = manifest.get("validated") and all(splits.get(split, {}).get("images") == count
                        for split, count in (("train", 239), ("val", 200), ("test", 779)))
            lines.append(checked(ready, task.upper() + " ReasonSeg：官方划分、独立副本、原生 mask/派生 bbox 与空目标支持", "准备中"))
        for job in reasonseg_jobs:
            run = Path(job["run_dir"])
            name = "Valen-Decision-" + {"det": "Det", "seg": "Seg"}[job["task"]] + "-" + job["size"]
            if job.get("variant") == "full":
                name += "-Full"
            elif job.get("variant") == "rlcd":
                name += "-RLCD" if job.get("initialization_variant") == "lora_merged" else "-Full-RLCD"
            checks = by_name[name]["checks"]
            completed = checks["reasonseg"] and checks["reasonseg_model_identity"]
            status = "等待补充评测" if (run / "pipeline_complete.json").is_file() else "等待模型完成"
            lines.append(checked(completed, name + "：ReasonSeg val / test 补充评测", status))
    if state.get("groundingme_support"):
        preflight = read_json(data_root / "eval/det/groundingme/manifests/preflight.json")
        lines.extend(["", "## 增补：GroundingME", "",
            checked(preflight.get("validated") is True and preflight.get("samples") == 1005,
                "官方 test 1,005 条：独立媒体、固定原始指令、正/空目标统一推理及训练集哈希隔离", "准备中")])
        for job in pipelines:
            if job["task"] != "det":
                continue
            name = "Valen-Decision-Det-" + job["size"] + ("-RLCD" if job.get("variant") == "rlcd" and job.get("initialization_variant") == "lora_merged" else "-Full-RLCD" if job.get("variant") == "rlcd" else "-Full" if job.get("variant") == "full" else "")
            checks = by_name[name]["checks"]
            lines.append(checked(checks.get("groundingme") and checks.get("groundingme_model_identity"),
                name + "：完整 test 总 Acc@0.5 与四维度/子类分数", "等待/进行补充评测"))
    if state.get("det_rlcd_plan"):
        active_sizes=state["det_rlcd_plan"].get("active_sizes")
        exclusive=active_sizes==["2B"]
        serial=state["det_rlcd_plan"].get("serial_order")
        merged = state["det_rlcd_plan"].get("initialization_variant") == "lora_merged"
        parent = "合并LoRA" if merged else "Full"
        tuning = "LoRA" if state["det_rlcd_plan"].get("rl_finetuning_type") == "lora" else "全参数"
        barrier=next((b for b in state.get("evaluation_barriers", []) if b.get("name")==(
            "det-first-three-lora" if merged else "det2-full64-vs-lora")), {})
        lines.extend(["", f"## 后续：Det {parent} 难例筛选与{tuning} RLCD", "",
            checked(barrier.get("ready") is True if merged or exclusive or serial else False,
                    "三组合并LoRA已验收；64卡按2B → 0.8B → 4B串行，每组完成11项评测后再接续下一组" if serial else
                    "2B合并LoRA完成验收；重新回放原100k筛选LoRA错例；其他尺寸暂停" if merged else
                    "2B Full与LoRA全部11项评测及性能/成本对照验收后，先启动2B RLCD；其他尺寸暂停" if exclusive else
                    "三组 Full 完成训练、导出及11项完整基线评测后，启用 RLCD 阶段", "等待合并LoRA验收" if merged else "等待2B Full完整对照" if exclusive else "等待三组 Full"),
            checked(bool(state["det_rlcd_plan"].get("implementation_ready")),
                    f"实现空间动作/空动作RLCD、固定参考KL及校准奖励；通过概率、{tuning}梯度/冻结范围及恢复测试", "待办")])
        for size in serial or state["det_rlcd_plan"]["sizes"]:
            name = "Valen-Decision-Det-" + size + ("-RLCD" if merged else "-Full-RLCD")
            resource=state["det_rlcd_plan"].get("per_size_training", {}).get(size, {"gpus":16,"global_batch":32})
            rl_job=next((j for j in pipelines if j.get("variant")=="rlcd" and j["size"]==size),None)
            recorded=by_name.get(name,{})
            hard=read_json(Path(rl_job["subset_dir"])/"manifest.json") if rl_job else {}
            replay_done=hard.get("validated") is True and hard.get("original_records")==100000
            training=recorded.get("joint",{})
            detail=(f"：{training['seen']:,}/{training['records']:,} 条，step {training['step']}" if training.get("records") else "")
            lines.extend([checked(recorded.get("deliverables_ready",False),name,
                                  "暂停，2B优先" if exclusive and size!="2B" else
                                  "进行中" if rl_job and rl_job.get("state")=="running" else
                                  f"等待 {rl_job['after_evaluated_model']} 完整评测验收" if rl_job and rl_job.get("after_evaluated_model") else
                                  f"等待 {parent} 基线"),
                checked(replay_done,f"以对应{parent}回放原100k，重新筛选并完整保存错误数量和模型/数据身份","待办","  "),
                checked(replay_done,"错误≤20k全部使用（低于10k也全取）；>20k均衡采20k，不复制凑数"
                        +(f"；错误 {hard['errors']:,}，选取 {hard['selected']:,}" if replay_done else ""),"待办","  "),
                checked(training.get("complete",False),f"从对应{parent}进行{tuning} RLCD：初始1epoch、{resource['gpus']}卡、全局batch{resource['global_batch']}；固定参考同为{parent}，保留基线"+detail,"进行中" if training.get("seen") else "待办","  "),
                checked(recorded.get("deliverables_ready",False),"完整模型导出/独立重载/示例；复测全部11项，对比LoRA/RLCD；已有Full结果同时列出","待办","  ")])
    if state.get("jev_api_evaluation"):
        api = state["jev_api_evaluation"]
        plan = read_json(Path(api["plan"]))
        status = read_json(Path(api["output_dir"]) / "status.json")
        metrics = read_json(Path(api["output_dir"]) / "metrics.json")
        benchmarks = metrics.get("benchmarks", {})
        native_complete = (metrics.get("complete_full_benchmarks") is True and metrics.get("model") == api["model"]
                    and set(benchmarks) == set(api["benchmarks"])
                    and sum(row.get("samples", 0) for row in benchmarks.values()) == api["samples"])
        report_root = Path(api.get("report_dir", Path(api["output_dir"]).parent / "reports"))
        report_path = report_root / "comparison.json"
        report, marker = read_json(report_path), read_json(report_root / "pipeline_complete.json")
        config_path = Path(api["output_dir"]) / "request-config.json"
        config = read_json(config_path)
        matched_report = (report.get("model") == api["model"] and config_path.is_file()
                          and report.get("request_config_sha256") == file_sha256(config_path))
        verified = {}
        for name in api["benchmarks"]:
            source = config.get("benchmarks", {}).get(name, {})
            metric = report.get("jev_benchmarks", {}).get(name, {})
            proof = report.get("prediction_verification", {}).get(name, {})
            expected = source.get("full_samples", 0)
            if (matched_report and expected > 0 and source.get("selected_samples") == expected
                    and metric.get("complete_split") is True
                    and metric.get("samples") == metric.get("full_samples") == proof.get("samples") == expected
                    and bool(source.get("records_sha256")) and metric.get("data_sha256") == source["records_sha256"]
                    and proof.get("all_expected_ids_verified") is True
                    and proof.get("original_box_iou_recomputed") is True):
                verified[name] = metric
        complete = (native_complete and set(verified) == set(api["benchmarks"])
                    and report.get("complete_full_evaluation") is True and marker.get("model") == api["model"]
                    and marker.get("samples") == api["samples"] and marker.get("split_count") == len(api["benchmarks"])
                    and marker.get("all_prediction_ids_verified") is True and report_path.is_file()
                    and marker.get("comparison_sha256") == file_sha256(report_path))
        count = status.get("predictions_completed", 0)
        lines.extend(["", "## Jev API Det 评测", "",
            checked(plan.get("key_validity") == "verified_http_200", "Jev官方API真实图片调用验证：HTTP 200"),
            checked(plan.get("pilot_validated") is True, "完整11项各4条试评测通过，共44条（不作为正式成绩）"),
            checked(complete, f"{api['model']}：完整{len(api['benchmarks'])}项、{api['samples']:,}条；已保存{count:,}条预测",
                    "等待逐ID复核" if native_complete else "后台评测中" if status.get("status") == "running" else "等待/检查评测进程")])
        for name in api["benchmarks"]:
            if name in verified:
                row = verified[name]
                detail = f"{row['samples']:,}条，Acc@0.5 {100*row['accuracy']:.2f}%（逐ID及原图IoU已核验）"
            else:
                expected = config.get("benchmarks", {}).get(name, {}).get("full_samples")
                saved = status.get("split_predictions", {}).get(name, benchmarks.get(name, {}).get("samples", 0))
                detail = f"{saved:,}/{expected:,}条，等待完整划分与逐ID复核" if expected else "等待完整划分与逐ID复核"
            lines.append(checked(name in verified, name + "：" + detail, "待完成", "  "))
    for title, jobs in (("原定模型：LoRA", originals), ("新增对照：第二阶段全量微调", full)):
        lines.extend(["", "## " + title, ""])
        for job in sorted(jobs, key=lambda item: (("det", "seg", "gui").index(item["task"]),
                                                ("0.8B", "2B", "4B", "9B").index(item["size"]))):
            run = Path(job["run_dir"])
            name = "Valen-Decision-" + {"det": "Det", "seg": "Seg", "gui": "GUI"}[job["task"]] + "-" + job["size"]
            variant = job.get("variant") == "full"
            if variant:
                name += "-Full"
            recorded = by_name[name]
            pipeline_complete = (run / "pipeline_complete.json").is_file()
            illustrated = recorded["checks"]["inference_examples"]
            complete = recorded["deliverables_ready"]
            status = {"running": "进行中", "queued": "排队中", "waiting_for_data": "等待数据",
                      "paused": "用户暂停，保留 checkpoint",
                      "waiting_for_dependencies": "等待前序依赖", "queued_in_allocation": "等待卡组前序任务",
                      "failed": "需处理", "blocked": "需处理"}.get(job.get("state"), "待办")
            if job.get("state") == "paused" and not (run / "joint/metrics.jsonl").is_file() and not (run / "warmup/metrics.jsonl").is_file():
                status = "用户暂停，尚未训练"
            deferred = bool(job.get("det_priority_requires")) and any(
                not Path(path).is_file() for path in job["det_priority_requires"])
            if deferred:
                status = "暂缓，Det 优先（保留已有 checkpoint）"
            if pipeline_complete and not complete:
                status = ("等待推理示例" if not illustrated and all(
                    value for key, value in recorded["checks"].items() if key != "inference_examples")
                    else "待补齐验收项")
            lines.append(checked(complete, name, status))
            if variant:
                identity = Path(job["baseline_run"]) / "comparison-init/initialization.json"
                lines.append(checked(identity.is_file(), "复用与 LoRA 相同的第一阶段最佳任务头", indent="  "))
                warmup_done = identity.is_file()
            else:
                warmup_done, detail, activity = stage_progress(run, "warmup", False, job.get("state"), deferred, job.get("training_samples", 100000))
                detail = detail.replace("固定 8 卡", f"固定 {job.get('gpus', 8)} 卡")
                lines.append(checked(warmup_done, "第一阶段任务头预热：" + detail, activity, "  "))
            joint_done, detail, activity = stage_progress(run, "joint", warmup_done, job.get("state"), deferred, job.get("training_samples", 100000))
            detail = detail.replace("固定 8 卡", f"固定 {job.get('gpus', 8)} 卡")
            lines.append(checked(joint_done, "第二阶段" + ("全量微调" if variant else "LoRA") + "：" + detail, activity, "  "))
            if deferred and job.get("pause_checkpoint"):
                saved = job["pause_checkpoint"]
                lines.append(checked(True, f"保留可恢复 checkpoint：{saved['seen']:,}/100,000 条，step {saved['step']:,}（含优化器与八卡 RNG）", indent="  "))
            export_done = all(recorded["checks"][key] for key in (
                "model_identity", "independent_export_reload", "export_files_present"))
            lines.append(checked(export_done, "完整模型导出、预测一致性与独立重载验证", indent="  "))
            expected = required_benchmarks(job["task"], job.get("coco_protocol", "standard"))
            bench_done = (recorded["checks"]["required_benchmarks"]
                          and recorded["checks"]["evaluation_model_identity"])
            lines.append(checked(bench_done, f"全部 {len(expected)} 个必测 benchmark/split 完整评测", indent="  "))
            lines.append(checked(illustrated, "保存固定样本的推理结果及少量可视化示例", indent="  "))
            if variant:
                lines.append(checked(recorded["checks"]["full_comparison"], "生成 LoRA/Full 指标差值、训练成本与推理成本对照", indent="  "))
    all_complete = results["deliverables_ready"]
    results_dir = Path(state.get("results_dir", control / "results"))
    archive = read_json(results_dir / "summary.json")
    results_done = (all_complete and not state.get("summary_error") and archive.get("deliverables_ready") is True
                    and archive.get("completed_models") == len(pipelines) and (results_dir / "summary.md").is_file())
    cleaned = all_complete and all(not (data_root / task / "raw").exists() for task in ("det", "seg", "gui"))
    cleaned = cleaned and all(not (Path(job["run_dir"]) / stage / name / "checkpoint.pt").exists()
                              for job in pipelines for stage in ("warmup", "joint") for name in ("best", "latest"))
    cleaned = cleaned and not (data_root / "eval/gui/raw").exists()
    cleaned = cleaned and all(not (Path(job["baseline_run"]) / "comparison-init/checkpoint.pt").exists() for job in full)
    lines.extend(["", "## 收尾验收", "",
        checked(all_complete, "完成全部模型与对照实验的训练、导出及完整评测"),
        checked(results_done, "归档各模型最终指标、耗时、显存及 LoRA/Full 对照结果"),
        checked(cleaned, "清理冗余 checkpoint、优化器状态和转换缓存；保留必要模型、数据及记录"), ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_progress(read_json(Path(args.registry)), args.registry, args.data_root, args.model_root, args.output)


if __name__ == "__main__":
    main()
