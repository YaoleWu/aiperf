# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import (
    MetricFlags,
    MetricSizeUnit,
    MetricTimeUnit,
    VideoJobStatus,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.common.models.record_models import VideoResponseData
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


def _video_job_boundaries(record: ParsedResponseRecord) -> tuple[int, int]:
    """Return the submit acknowledgement and observed completion timestamps."""
    if len(record.responses) < 2:
        raise NoMetricValue("Video submit and completion responses are required.")
    submit, completed = record.responses[0], record.responses[-1]
    if (
        not isinstance(submit.data, VideoResponseData)
        or not isinstance(completed.data, VideoResponseData)
        or completed.data.status != VideoJobStatus.COMPLETED
        or not record.start_perf_ns <= submit.perf_ns <= completed.perf_ns
    ):
        raise NoMetricValue(
            "Ordered video submit and completion responses are required."
        )
    return submit.perf_ns, completed.perf_ns


class VideoJobSubmitLatencyMetric(BaseRecordMetric[int]):
    """Client time from request start to the video job acknowledgement."""

    tag = "video_job_submit_latency"
    header = "Video Job Submit Latency"
    short_header = "Video Submit"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 306
    flags = MetricFlags.PRODUCES_VIDEO_ONLY

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        submit_ns, _ = _video_job_boundaries(record)
        return submit_ns - record.start_perf_ns


class VideoJobPollWaitMetric(BaseRecordMetric[int]):
    """Client time from submit acknowledgement to observed job completion."""

    tag = "video_job_poll_wait"
    header = "Video Job Poll Wait"
    short_header = "Video Poll Wait"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 307
    flags = MetricFlags.PRODUCES_VIDEO_ONLY

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        submit_ns, completed_ns = _video_job_boundaries(record)
        return completed_ns - submit_ns


class VideoContentDownloadLatencyMetric(BaseRecordMetric[int]):
    """Client time from observed completion to a successful content download."""

    tag = "video_content_download_latency"
    header = "Video Content Download Latency"
    short_header = "Video Download"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 308
    flags = MetricFlags.PRODUCES_VIDEO_ONLY

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        _, completed_ns = _video_job_boundaries(record)
        download_ns = record.request.video_download_end_perf_ns
        if download_ns is None or download_ns < completed_ns:
            raise NoMetricValue("No successful video content download.")
        return download_ns - completed_ns


class VideoJobE2ELatencyMetric(BaseRecordMetric[int]):
    """Client time to observed completion, including optional content download."""

    tag = "video_job_e2e_latency"
    header = "Video Job E2E Latency"
    short_header = "Video Job E2E"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 309
    flags = MetricFlags.PRODUCES_VIDEO_ONLY

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        _, completed_ns = _video_job_boundaries(record)
        download_ns = record.request.video_download_end_perf_ns
        if download_ns is not None and download_ns < completed_ns:
            raise NoMetricValue("Video download timestamp precedes job completion.")
        end_ns = completed_ns if download_ns is None else download_ns
        return end_ns - record.start_perf_ns


class VideoInferenceTimeMetric(BaseRecordMetric[float]):
    """Server-reported video inference time from SGLang.

    Extracts the `inference_time_s` field from the video generation response,
    which represents the actual GPU generation time reported by the server.
    """

    tag = "video_inference_time"
    header = "Video Inference Time"
    short_header = "Inference Time"
    unit = MetricTimeUnit.SECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 310  # After request_latency (300)
    flags = MetricFlags.PRODUCES_VIDEO_ONLY

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """Extract inference_time_s from VideoResponseData."""
        for response in record.responses:
            if (
                isinstance(response.data, VideoResponseData)
                and response.data.inference_time_s is not None
            ):
                return response.data.inference_time_s

        raise NoMetricValue("Video inference time not available in response.")


class VideoPeakMemoryMetric(BaseRecordMetric[float]):
    """Server-reported peak GPU memory usage from SGLang.

    Extracts the `peak_memory_mb` field from the video generation response,
    which represents the peak GPU memory used during generation.
    """

    tag = "video_peak_memory"
    header = "Video Peak Memory"
    short_header = "Peak Memory"
    unit = MetricSizeUnit.MEGABYTES
    display_order = 311  # After video_inference_time (310)
    flags = MetricFlags.PRODUCES_VIDEO_ONLY

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> float:
        """Extract peak_memory_mb from VideoResponseData."""
        for response in record.responses:
            if (
                isinstance(response.data, VideoResponseData)
                and response.data.peak_memory_mb is not None
            ):
                return response.data.peak_memory_mb

        raise NoMetricValue("Video peak memory not available in response.")
