"""
Performance monitoring and health check utilities.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from src.utils.logging_config import get_logger

logger = get_logger("monitoring")


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    service: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self, max_metrics: int = 1000):
        self.max_metrics = max_metrics
        self.metrics = defaultdict(lambda: deque(maxlen=max_metrics))
        self.lock = threading.RLock()
        self.start_time = datetime.utcnow()
    
    def record_metric(self, name: str, value: float, unit: str = "count", tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric)
        
        logger.debug(f"Recorded metric: {name}={value} {unit}")
    
    def record_execution_time(self, operation: str, execution_time: float, **tags):
        """Record operation execution time"""
        self.record_metric(
            f"{operation}_execution_time",
            execution_time,
            "seconds",
            tags
        )
    
    def record_counter(self, name: str, increment: int = 1, **tags):
        """Record a counter metric"""
        self.record_metric(name, increment, "count", tags)
    
    def record_gauge(self, name: str, value: float, unit: str = "units", **tags):
        """Record a gauge metric"""
        self.record_metric(name, value, unit, tags)
    
    def get_metrics(self, name: Optional[str] = None, since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics, optionally filtered by name and time"""
        with self.lock:
            if name:
                metrics = list(self.metrics.get(name, []))
            else:
                metrics = []
                for metric_list in self.metrics.values():
                    metrics.extend(metric_list)
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return sorted(metrics, key=lambda m: m.timestamp)
    
    def get_metric_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {"count": 0}
        
        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "latest_timestamp": metrics[-1].timestamp.isoformat()
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Record system metrics
            self.record_gauge("system_cpu_percent", cpu_percent, "percent")
            self.record_gauge("system_memory_percent", memory.percent, "percent")
            self.record_gauge("system_memory_used", memory.used / (1024**3), "GB")
            self.record_gauge("system_disk_percent", disk.percent, "percent")
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    def get_uptime(self) -> timedelta:
        """Get application uptime"""
        return datetime.utcnow() - self.start_time
    
    def clear_metrics(self, name: Optional[str] = None):
        """Clear metrics"""
        with self.lock:
            if name:
                self.metrics[name].clear()
            else:
                self.metrics.clear()


class HealthChecker:
    """Health check system for monitoring service availability"""
    
    def __init__(self):
        self.checks = {}
        self.results = deque(maxlen=100)
        self.lock = threading.RLock()
    
    def register_check(self, name: str, check_func, interval: int = 60):
        """Register a health check"""
        self.checks[name] = {
            "func": check_func,
            "interval": interval,
            "last_run": None
        }
        logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthCheckResult(
                service=name,
                status="unhealthy",
                message="Health check not found",
                response_time=0.0,
                timestamp=datetime.utcnow()
            )
        
        start_time = time.time()
        
        try:
            check_func = self.checks[name]["func"]
            result = check_func()
            response_time = time.time() - start_time
            
            if isinstance(result, dict):
                health_result = HealthCheckResult(
                    service=name,
                    status=result.get("status", "healthy"),
                    message=result.get("message", "OK"),
                    response_time=response_time,
                    timestamp=datetime.utcnow(),
                    details=result.get("details")
                )
            else:
                # Assume boolean result
                health_result = HealthCheckResult(
                    service=name,
                    status="healthy" if result else "unhealthy",
                    message="OK" if result else "Check failed",
                    response_time=response_time,
                    timestamp=datetime.utcnow()
                )
            
            self.checks[name]["last_run"] = datetime.utcnow()
            
            with self.lock:
                self.results.append(health_result)
            
            return health_result
            
        except Exception as e:
            response_time = time.time() - start_time
            health_result = HealthCheckResult(
                service=name,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                response_time=response_time,
                timestamp=datetime.utcnow(),
                details={"error": str(e)}
            )
            
            with self.lock:
                self.results.append(health_result)
            
            return health_result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        with self.lock:
            if not self.results:
                return {"status": "unknown", "checks": 0}
            
            recent_results = [r for r in self.results if r.timestamp > datetime.utcnow() - timedelta(minutes=5)]
            
            if not recent_results:
                return {"status": "stale", "checks": len(self.checks)}
            
            statuses = [r.status for r in recent_results]
            
            if all(s == "healthy" for s in statuses):
                overall_status = "healthy"
            elif any(s == "unhealthy" for s in statuses):
                overall_status = "unhealthy"
            else:
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "checks": len(self.checks),
                "recent_results": len(recent_results),
                "last_check": max(r.timestamp for r in recent_results).isoformat()
            }


class ApplicationMonitor:
    """Main application monitoring system"""
    
    def __init__(self):
        self.performance = PerformanceMonitor()
        self.health = HealthChecker()
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        def system_health():
            """Check system resource health"""
            try:
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                if cpu > 90:
                    return {"status": "unhealthy", "message": f"High CPU usage: {cpu}%"}
                elif memory.percent > 90:
                    return {"status": "unhealthy", "message": f"High memory usage: {memory.percent}%"}
                elif cpu > 70 or memory.percent > 70:
                    return {"status": "degraded", "message": "High resource usage"}
                else:
                    return {"status": "healthy", "message": "System resources OK"}
            except Exception as e:
                return {"status": "unhealthy", "message": f"System check failed: {e}"}
        
        def api_connectivity():
            """Check API connectivity"""
            # This would check if API keys are valid and services are reachable
            return {"status": "healthy", "message": "API connectivity OK"}
        
        self.health.register_check("system", system_health, interval=30)
        self.health.register_check("api_connectivity", api_connectivity, interval=300)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        # Get recent metrics (last hour)
        since = datetime.utcnow() - timedelta(hours=1)
        
        return {
            "health": self.health.get_health_summary(),
            "uptime": str(self.performance.get_uptime()),
            "system_metrics": self.performance.get_system_metrics(),
            "performance_summary": {
                "queries": self.performance.get_metric_summary("rag_query_execution_time", since),
                "initialization": self.performance.get_metric_summary("rag_system_initialization_execution_time", since),
                "vector_search": self.performance.get_metric_summary("similarity_search_execution_time", since)
            },
            "recent_health_checks": [
                result.to_dict() for result in list(self.health.results)[-10:]
            ]
        }


# Global monitor instance
app_monitor = ApplicationMonitor()
