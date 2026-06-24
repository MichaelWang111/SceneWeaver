测试完了，结果和你想验证的清单基本一致。

**结果**
- `apply_patch` 添加成功：`sandbox_probe/apply_patch_probe.txt`
- `apply_patch` 更新成功：内容从 `apply_patch add: ok` 改成 `apply_patch update: ok`
- `apply_patch` 删除成功：`sandbox_probe/delete_probe.txt` 先添加后单独删除成功
- 正常工作区 PowerShell 写入成功：`sandbox_probe/powershell_workspace_probe.txt`
- 临时目录 PowerShell 写入成功：`$env:TEMP\codex_sandbox_tmp_probe.txt`
- 工作区外 PowerShell 写入被阻止：
  ```text
  outside write: denied
  Access to the path 'C:\Users\Michael\codex_outside_workspace_probe.txt' is denied.
  ```

有一个额外发现：如果在同一个 `apply_patch_batch` 里“添加一个新文件，然后立刻删除同一个新文件”，会失败：

```text
Failed to read ...\sandbox_probe\delete_probe.txt: 系统找不到指定的文件。
```

这看起来是批量补丁的删除校验顺序问题，不是删除能力本身的问题。分成两步后，删除正常成功。

当前留下的测试文件是：

- [sandbox_probe/apply_patch_probe.txt](D:/Documents/测试一下/sandbox_probe/apply_patch_probe.txt)
- [sandbox_probe/powershell_workspace_probe.txt](D:/Documents/测试一下/sandbox_probe/powershell_workspace_probe.txt)

`git status --short` 显示：

```text
?? sandbox_probe/
```