# GitHub 发布与持续更新

此仓库采用default-deny `.gitignore`，只发布四个当前代码包、根README和同步脚本。正式数据、结果、checkpoint和凭据不会进入Git索引。

## 首次远程连接

远程仓库应默认创建为private。配置`origin`后执行：

```bash
bash github_sync/update_snapshot.sh
```

## 持续更新

一次性增量同步：

```bash
bash github_sync/update_snapshot.sh
```

持续监控模式默认每30分钟检查一次；只有跟踪文件发生变化时才提交和推送：

```bash
bash github_sync/watch_and_push.sh 1800
```

脚本不会执行`git add -f`，不会越过`.gitignore`。如果没有`origin`或推送失败，会返回非零并保留本地commit，不伪装成远程已更新。
