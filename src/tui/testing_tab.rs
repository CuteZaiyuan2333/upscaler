// 测试选项卡 UI 渲染。

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::state::{RunStatus, TestField, TestingMode, TestingTabState};
use super::ui::{render_button, render_input_field, render_progress_bar};

pub fn draw(frame: &mut Frame, area: Rect, state: &TestingTabState) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);

    draw_config_panel(frame, columns[0], state);
    draw_progress_panel(frame, columns[1], state);
}

fn draw_config_panel(frame: &mut Frame, area: Rect, state: &TestingTabState) {
    let block = Block::default()
        .title(" Testing Configuration ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // 模式选择
    let mode_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Mode label
            Constraint::Length(1), // Mode switch
            Constraint::Length(1), // spacing
        ])
        .split(inner);

    frame.render_widget(
        Line::from(" Testing Mode:").style(Style::default().fg(Color::White)),
        mode_rows[0],
    );

    let is_auto = state.mode == TestingMode::AutoTest;
    let auto_style = if is_auto {
        Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };
    let direct_style = if !is_auto {
        Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };

    let cursor = if state.focused_field == TestField::ModeSwitch {
        Span::styled("▶ ", Style::default().fg(Color::Cyan))
    } else {
        Span::raw("  ")
    };

    let mode_line = Line::from(vec![
        cursor,
        Span::styled("[ Auto Test ]", auto_style),
        Span::raw("  "),
        Span::styled("[ Direct Inference ]", direct_style),
    ]);
    frame.render_widget(mode_line, mode_rows[1]);

    // 剩余字段区域
    let fields_area = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(inner)[1];

    match state.mode {
        TestingMode::AutoTest => draw_auto_test_fields(frame, fields_area, state),
        TestingMode::DirectInference => draw_direct_inference_fields(frame, fields_area, state),
    }
}

fn draw_auto_test_fields(frame: &mut Frame, area: Rect, state: &TestingTabState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Test Image
            Constraint::Length(1), // Checkpoint
            Constraint::Length(1), // Output Dir
            Constraint::Length(1), // spacing
            Constraint::Length(3), // button
        ])
        .split(area);

    render_input_field(
        "Test Image (1024x1024)",
        &state.test_image,
        state.focused_field == TestField::TestImage,
        false,
        rows[0],
        frame,
    );
    render_input_field(
        "Checkpoint Dir/File",
        &state.checkpoint_path,
        state.focused_field == TestField::CheckpointPath,
        false,
        rows[1],
        frame,
    );
    render_input_field(
        "Output Dir",
        &state.output_dir,
        state.focused_field == TestField::OutputDir,
        false,
        rows[2],
        frame,
    );

    let button_label = match state.status {
        RunStatus::Idle => "▶ Run Auto Test",
        RunStatus::Running => "● Testing...",
        RunStatus::Completed => "✓ Complete",
    };
    render_button(
        button_label,
        state.focused_field == TestField::StartButton,
        rows[4],
        frame,
    );
}

fn draw_direct_inference_fields(frame: &mut Frame, area: Rect, state: &TestingTabState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Main Image
            Constraint::Length(1), // Reference Image
            Constraint::Length(1), // Checkpoint
            Constraint::Length(1), // Output Path
            Constraint::Length(1), // spacing
            Constraint::Length(3), // button
        ])
        .split(area);

    render_input_field(
        "Main Image (HxW)",
        &state.main_image,
        state.focused_field == TestField::MainImage,
        false,
        rows[0],
        frame,
    );
    render_input_field(
        "Reference Image (4Hx4W)",
        &state.reference_image,
        state.focused_field == TestField::ReferenceImage,
        false,
        rows[1],
        frame,
    );
    render_input_field(
        "Checkpoint File",
        &state.checkpoint_path,
        state.focused_field == TestField::CheckpointPath,
        false,
        rows[2],
        frame,
    );
    render_input_field(
        "Output Path",
        &state.output_path,
        state.focused_field == TestField::OutputPath,
        false,
        rows[3],
        frame,
    );

    let button_label = match state.status {
        RunStatus::Idle => "▶ Run Inference",
        RunStatus::Running => "● Running...",
        RunStatus::Completed => "✓ Complete",
    };
    render_button(
        button_label,
        state.focused_field == TestField::StartButton,
        rows[5],
        frame,
    );
}

fn draw_progress_panel(frame: &mut Frame, area: Rect, state: &TestingTabState) {
    let block = Block::default()
        .title(" Test Progress ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Status
            Constraint::Length(1), // Mode
            Constraint::Length(1), // Current
            Constraint::Length(1), // Progress bar
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

    let mode_str = match state.mode {
        TestingMode::AutoTest => "Auto Test (all checkpoints × 16 filters)",
        TestingMode::DirectInference => "Direct Inference (no filters)",
    };
    frame.render_widget(
        Line::from(format!(" Mode: {}", mode_str)),
        rows[1],
    );

    frame.render_widget(
        Line::from(format!(" Current: {}", state.current_item)),
        rows[2],
    );

    render_progress_bar(state.completed, state.total, "Progress", rows[3], frame);

    // 日志
    let log_block = Block::default()
        .title(" Log ")
        .borders(Borders::ALL);
    let log_inner = log_block.inner(rows[5]);
    frame.render_widget(log_block, rows[5]);

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