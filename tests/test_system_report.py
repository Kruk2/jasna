import logging

from jasna import system_report


def test_log_system_summary_logs_a_warning_instead_of_raising(monkeypatch, caplog):
    def unreadable_cpu():
        raise OSError("no cpuinfo")

    monkeypatch.setattr(system_report, "_cpu_name", unreadable_cpu)
    system_report.log_system_summary.cache_clear()
    with caplog.at_level(logging.WARNING, logger=system_report.__name__):
        system_report.log_system_summary()
    system_report.log_system_summary.cache_clear()

    assert "Could not read the system summary: no cpuinfo" in caplog.text
