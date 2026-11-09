use std::time::Duration;

/// Holds execution and latency metrics collected locally by a worker thread.
#[derive(Default, Debug, Clone, Copy)]
pub struct StageMetrics {
    pub busy: Duration,
    pub idle: Duration,
    pub items: u64,
}

impl StageMetrics {
    /// Percentage of total lifetime spent doing active compute work (0.0% to 100.0%).
    pub fn utilization(&self) -> f64 {
        let total = self.busy.as_secs_f64() + self.idle.as_secs_f64();
        if total > 0.0 {
            (self.busy.as_secs_f64() / total) * 100.0
        } else {
            0.0
        }
    }
}

/// Formats and prints a pipeline health and bottleneck summary table.
pub fn print_pipeline_telemetry(stages: &[(&str, StageMetrics)]) {
    eprintln!(
        "\n╔════════════════════════════════════════════════════════════════════════════════╗"
    );
    eprintln!("║                           PIPELINE TELEMETRY REPORT                            ║");
    eprintln!("╠══════════════════════╦══════════════╦══════════════╦═════════════╦═════════════╣");
    eprintln!("║ Stage                ║ Busy Time    ║ Idle (Wait)  ║ Utilization ║ Throughput  ║");
    eprintln!("╠══════════════════════╬══════════════╬══════════════╬═════════════╬═════════════╣");

    for (name, m) in stages {
        let total_secs = m.busy.as_secs_f64() + m.idle.as_secs_f64();
        let rate = if total_secs > 0.0 {
            m.items as f64 / total_secs
        } else {
            0.0
        };

        let f_name = if name.chars().count() > 20 {
            format!("{}...", name.chars().take(17).collect::<String>())
        } else {
            name.to_string()
        };

        let f_busy = format!("{:.2?}", m.busy);
        let f_idle = format!("{:.2?}", m.idle);
        let f_util = format!("{:.1}%", m.utilization());
        let f_rate = format!("{rate:.0}/s");

        eprintln!(
            "║ {f_name:<20.20} ║ {f_busy:>12.12} ║ {f_idle:>12.12} ║ {f_util:>11.11} ║ {f_rate:>11.11} ║",
        );
    }

    eprintln!("╚══════════════════════╩══════════════╩══════════════╩═════════════╩═════════════╝");
}
