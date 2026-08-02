// 训练选项卡 UI 渲染。

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::state::{RunStatus, TrainField, TrainingTabState};
use super::ui::{render_button, render_input_field, render_toggle};

pub fn draw(frame: &mut Frame, area: Rect, state: &TrainingTabState) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);

    draw_config_panel(frame, columns[0], state);
    draw_progress_panel(frame, columns[1], state);
}

fn draw_config_panel(frame: &mut Frame, area: Rect, state: &TrainingTabState) {
    let block = Block::default()
        .title(" Training Configuration ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // 字段列表
    let fields: &[(&str, &str, TrainField)] = &[
        ("Checkpoint Dir", &state.checkpoint_dir, TrainField::CheckpointDir),
        ("Resume Ckpt", &state.resume_checkpoint, TrainField::ResumeCheckpoint),
        ("Crop Size", &state.crop_size, TrainField::CropSize),
        ("Dataset Dir", &state.dataset_dir, TrainField::DatasetDir),
        ("Steps/Epoch", &state.steps_per_epoch, TrainField::StepsPerEpoch),
        ("Epochs", &state.num_epochs, TrainField::Epochs),
        ("Learning Rate", &state.learning_rate, TrainField::LearningRate),
        ("Weight Decay", &state.weight_decay, TrainField::WeightDecay),
        ("Ref Noise", &state.ref_noise, TrainField::RefNoise),
        ("Seed", &state.seed, TrainField::Seed),
    ];

    let num_text_fields = fields.len();
    let num_toggles = 2;
    let num_button = 1;
    let _total_rows = num_text_fields + num_toggles + num_button + 2; // +2 for spacing

    let constraints: Vec<Constraint> = (0..num_text_fields)
        .map(|_| Constraint::Length(1))
        .chain((0..num_toggles).map(|_| Constraint::Length(1)))
        .chain(std::iter::once(Constraint::Length(1))) // spacing
        .chain(std::iter::once(Constraint::Length(3))) // button
        .collect();

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    for (i, (label, value, field)) in fields.iter().enumerate() {
        render_input_field(label, value, state.focused_field == *field, false, rows[i], frame);
    }

    // 开关
    let toggle_start = num_text_fields;
    render_toggle(
        "Full Resolution",
        state.full_resolution,
        state.focused_field == TrainField::FullResolution,
        rows[toggle_start],
        frame,
    );
    render_toggle(
        "Camera Filters",
        state.use_camera_filters,
        state.focused_field == TrainField::CameraFilters,
        rows[toggle_start + 1],
        frame,
    );

    // 按钮
    let button_label = match state.status {
        RunStatus::Idle => "▶ Start Training",
        RunStatus::Running => "● Training...",
        RunStatus::Completed => "✓ Complete",
    };
    render_button(
        button_label,
        state.focused_field == TrainField::StartButton,
        rows[toggle_start + 3],
        frame,
    );
}

fn draw_progress_panel(frame: &mut Frame, area: Rect, state: &TrainingTabState) {
    let block = Block::default()
        .title(" Training Progress ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Status
            Constraint::Length(1), // Epoch
            Constraint::Length(1), // Step
            Constraint::Length(1), // Loss
            Constraint::Length(1), // Avg Loss
            Constraint::Length(1), // Samples
            Constraint::Length(1), // empty
            Constraint::Min(1),    // Log
        ])
        .split(inner);

    let status_str = match state.status {
        RunStatus::Idle => "Idle",
        RunStatus::Running => "Running...",
        RunStatus::Completed => "Completed ✓",
    };
    frame.render_widget(
        Line::from(format!(" Status: {}", status_str)).style(Style::default().fg(Color::Yellow)),
        rows[0],
    );

    if state.dataset_size > 0 {
        frame.render_widget(
            Line::from(format!(" Dataset: {} images, Params: {}", state.dataset_size, state.num_params)),
            rows[1],
        );
    } else {
        frame.render_widget(
            Line::from(" Dataset: N/A"),
            rows[1],
        );
    }

    frame.render_widget(
        Line::from(format!(" Epoch: {} / {}", state.current_epoch, state.total_epochs)),
        rows[2],
    );

    frame.render_widget(
        Line::from(format!(" Global Step: {}", state.global_step)),
        rows[3],
    );

    frame.render_widget(
        Line::from(format!(" Loss: {:.6}", state.current_loss)),
        rows[4],
    );

    frame.render_widget(
        Line::from(format!(" Avg Loss: {:.6}", state.avg_loss)),
        rows[5],
    );

    frame.render_widget(
        Line::from(format!(" Samples: {}", state.samples_seen)),
        rows[6],
    );

    // 日志区域
    let log_block = Block::default()
        .title(" Log ")
        .borders(Borders::ALL);
    let log_inner = log_block.inner(rows[7]);
    frame.render_widget(log_block, rows[7]);

    let log_height = log_inner.height as usize;
    let start = if state.log.len() > log_height {
        state.log.len() - log_height
    } else {
        0
    };
    let log_text: Vec<Line> = state.log[start..]
        .iter()
        .map(|l| Line::from(l.as_str()))
        .collect();
    if !log_text.is_empty() {
        frame.render_widget(Paragraph::new(log_text), log_inner);
    }
}