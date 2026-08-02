// TUI 整体布局渲染。

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Tabs},
    Frame,
};

use super::state::{AppState, InputMode, Tab};

pub fn draw(frame: &mut Frame, state: &AppState) {
    let area = frame.area();

    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // title
            Constraint::Length(2),  // tabs
            Constraint::Min(1),     // content
            Constraint::Length(1),  // bottom bar
        ])
        .split(area);

    // 标题行
    let title = Line::from(vec![
        Span::styled(" RefGuidedUpsampler ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
    ]);
    frame.render_widget(title.centered(), main_chunks[0]);

    // 选项卡
    let tab_titles = vec![" Training ", " Testing ", " Settings "];
    let tabs = Tabs::new(tab_titles)
        .select(match state.active_tab {
            Tab::Training => 0,
            Tab::Testing => 1,
            Tab::Settings => 2,
        })
        .style(Style::default().fg(Color::Gray))
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .divider("│");
    frame.render_widget(tabs, main_chunks[1]);

    // 主内容区域
    let main_area = main_chunks[2];
    match state.active_tab {
        Tab::Training => super::training_tab::draw(frame, main_area, &state.training),
        Tab::Testing => super::testing_tab::draw(frame, main_area, &state.testing),
        Tab::Settings => super::settings_tab::draw(frame, main_area, &state.settings),
    }

    // 底部栏
    draw_bottom_bar(frame, main_chunks[3], state);
}

fn draw_bottom_bar(frame: &mut Frame, area: Rect, state: &AppState) {
    match state.input_mode {
        InputMode::Command => {
            let text = format!(" /{}", state.input_buffer);
            let line = Line::from(vec![
                Span::styled(text, Style::default().fg(Color::Yellow).bg(Color::DarkGray)),
                Span::styled(" ", Style::default().bg(Color::DarkGray)),
            ]);
            frame.render_widget(line, area);
        }
        InputMode::EditingField => {
            let label = &state.editing_field_label;
            let text = format!(" {}: {}", label, state.input_buffer);
            let line = Line::from(vec![
                Span::styled(text, Style::default().fg(Color::Black).bg(Color::Yellow)),
                Span::styled(" █", Style::default().fg(Color::Yellow).bg(Color::Yellow)),
            ]);
            frame.render_widget(line, area);
        }
        InputMode::Normal => {
            draw_status_bar(frame, area, state);
        }
    }
}

fn draw_status_bar(frame: &mut Frame, area: Rect, state: &AppState) {
    let text = match state.active_tab {
        Tab::Training => match state.training.status {
            super::state::RunStatus::Idle =>
                " ↑↓:Navigate  Enter:Edit  Space:Toggle  Tab:Switch  /:Command  /quit:Exit  |  Log: upscaler.log ",
            super::state::RunStatus::Running =>
                " Training...  |  Log: upscaler.log ",
            super::state::RunStatus::Completed =>
                " Training Complete  |  Log: upscaler.log ",
        },
        Tab::Testing => match state.testing.status {
            super::state::RunStatus::Idle =>
                " ↑↓:Navigate  ←→:Mode  Enter:Edit/Run  Space:Toggle  Tab:Switch  /:Command  /quit:Exit  |  Log: upscaler.log ",
            super::state::RunStatus::Running =>
                " Testing...  |  Log: upscaler.log ",
            super::state::RunStatus::Completed =>
                " Testing Complete  |  Log: upscaler.log ",
        },
        Tab::Settings =>
            " ↑↓:Navigate  ←→:Backend  Tab:Switch  /:Command  /quit:Exit  |  Log: upscaler.log ",
    };

    let line = Line::from(vec![
        Span::styled(text, Style::default().fg(Color::Black).bg(Color::DarkGray)),
    ]);
    frame.render_widget(line, area);
}

// 辅助：绘制输入字段
pub fn render_input_field(
    label: &str,
    value: &str,
    focused: bool,
    _editing: bool,
    area: Rect,
    frame: &mut Frame,
) {
    let style = if focused {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };

    let cursor = if focused { "▶" } else { " " };
    let display = format!(" {} {}: {}", cursor, label, if value.is_empty() { "(empty)" } else { value });
    let text = Line::from(vec![Span::styled(display, style)]);
    frame.render_widget(text, area);
}

// 辅助：绘制开关
pub fn render_toggle(
    label: &str,
    value: bool,
    focused: bool,
    area: Rect,
    frame: &mut Frame,
) {
    let style = if focused {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };

    let status = if value { "[■] ON " } else { "[ ] OFF" };
    let text = if focused {
        Line::from(vec![
            Span::styled("▶ ", Style::default().fg(Color::Cyan)),
            Span::styled(format!(" {}: {}", label, status), style),
        ])
    } else {
        Line::from(vec![
            Span::raw("  "),
            Span::styled(format!(" {}: {}", label, status), style),
        ])
    };
    frame.render_widget(text, area);
}

// 辅助：绘制按钮
pub fn render_button(
    label: &str,
    focused: bool,
    area: Rect,
    frame: &mut Frame,
) {
    let style = if focused {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };

    let text = format!(" [ {} ] ", label);
    let block = Block::default().style(style);
    let inner = block.inner(area);
    frame.render_widget(Paragraph::new(text).centered().style(style), inner);
    frame.render_widget(block, area);
}

// 辅助：绘制进度条
pub fn render_progress_bar(
    completed: usize,
    total: usize,
    label: &str,
    area: Rect,
    frame: &mut Frame,
) {
    let ratio = if total > 0 { completed as f64 / total as f64 } else { 0.0 };
    let bar_width = (area.width as usize).saturating_sub(4);
    let filled = (bar_width as f64 * ratio) as usize;
    let empty = bar_width.saturating_sub(filled);

    let bar = format!(
        "{}{} {}/{}",
        "█".repeat(filled),
        "░".repeat(empty),
        completed,
        total,
    );

    let text = if label.is_empty() { bar } else { format!("{}: {}", label, bar) };
    frame.render_widget(Paragraph::new(text).style(Style::default().fg(Color::Cyan)), area);
}