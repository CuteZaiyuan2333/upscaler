// 设置选项卡 — 后端选择等全局配置。

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::state::{SettingsField, SettingsTabState};

pub fn draw(frame: &mut Frame, area: Rect, state: &SettingsTabState) {
    let block = Block::default()
        .title(" Settings ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // info
            Constraint::Length(1), // spacing
            Constraint::Length(1), // training backend label
            Constraint::Length(1), // training backend value
            Constraint::Length(1), // spacing
            Constraint::Length(1), // inference backend label
            Constraint::Length(1), // inference backend value
            Constraint::Length(1), // spacing
            Constraint::Length(1), // note
            Constraint::Min(1),
        ])
        .split(inner);

    // 说明
    frame.render_widget(
        Paragraph::new(" Backend selection (requires corresponding Cargo feature: wgpu, cuda, ndarray)")
            .style(Style::default().fg(Color::DarkGray)),
        rows[0],
    );

    // 训练后端
    let train_focused = state.focused_field == SettingsField::TrainingBackend;
    let train_style = if train_focused {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };
    frame.render_widget(
        Line::from(vec![
            Span::styled(" Training Backend: ", train_style),
            Span::styled(
                format!("[ {} ]", state.training_backend.label()),
                if train_focused { Style::default().fg(Color::Black).bg(Color::Cyan) } else { Style::default().fg(Color::Cyan) },
            ),
            Span::raw("  ← → to change"),
        ]),
        rows[2],
    );
    frame.render_widget(
        Line::from("  (autodiff required — Wgpu or Cuda only)")
            .style(Style::default().fg(Color::DarkGray)),
        rows[3],
    );

    // 推理后端
    let inf_focused = state.focused_field == SettingsField::InferenceBackend;
    let inf_style = if inf_focused {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };
    frame.render_widget(
        Line::from(vec![
            Span::styled(" Inference Backend:", inf_style),
            Span::styled(
                format!("[ {} ]", state.inference_backend.label()),
                if inf_focused { Style::default().fg(Color::Black).bg(Color::Cyan) } else { Style::default().fg(Color::Cyan) },
            ),
            Span::raw("  ← → to change"),
        ]),
        rows[5],
    );
    frame.render_widget(
        Line::from("  (inference only — any backend works)")
            .style(Style::default().fg(Color::DarkGray)),
        rows[6],
    );

    // 备注
    frame.render_widget(
        Paragraph::new(" ↑↓: navigate  |  ←→: change backend  |  Tab: switch tab  |  /quit: exit")
            .style(Style::default().fg(Color::DarkGray)),
        rows[8],
    );
}