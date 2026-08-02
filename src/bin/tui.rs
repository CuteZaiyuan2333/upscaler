// Ratatui TUI 应用入口。
//
// 用法：
//   cargo run --release --bin tui
//
// 快捷键：
//   1/2 或 Tab  — 切换选项卡
//   ↑↓          — 导航表单字段
//   Enter       — 启动训练/测试
//   Space       — 切换开关
//   q / Esc     — 退出

use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use std::io::stdout;

use upscaler::tui::app::App;
use upscaler::tui::state::AppState;

fn main() -> Result<(), String> {
    // Windows: WGPU/DXC 编译器线程需要大栈，在进程级别设置默认栈
    unsafe { std::env::set_var("RUST_MIN_STACK", "33554432"); }

    // 设置终端
    let mut stdout = stdout();
    terminal::enable_raw_mode().map_err(|e| e.to_string())?;
    stdout
        .execute(EnterAlternateScreen)
        .map_err(|e| e.to_string())?;

    // 创建 terminal backend
    let terminal = ratatui::init();

    // 创建 app state
    let state = AppState::default();
    let mut app = App::new(state, terminal);

    // 运行事件循环
    let result = app.run();

    // 恢复终端
    ratatui::restore();
    terminal::disable_raw_mode().map_err(|e| e.to_string())?;
    stdout
        .execute(LeaveAlternateScreen)
        .map_err(|e| e.to_string())?;

    result
}